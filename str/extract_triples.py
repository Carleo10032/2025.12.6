"""
知识三元组抽取主程序

修改要点：
1. 导入提示词模块
2. 导入质量控制模块
3. 在工作流中使用这些模块
"""

import os
import json
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import click
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from tqdm import tqdm

# 🆕 导入提示词和质量控制模块
from prompts import TripleExtractionPrompts, PromptVersion
from triple_quality import TripleQualityController

# ==================== 数据模型 ====================

@dataclass
class TextBlock:
    """文本块数据模型"""
    text: str
    source_text_id: str
    source_file: str
    page_number: Optional[int] = None


# ==================== 工作流 ====================

class TripleExtractionWorkflow:
    """知识三元组抽取工作流 - 批量处理版"""
    
    def __init__(
        self,
        llm: LlamaOpenAI,
        prompt_version: str = "v3_academic",
        enable_quality_control: bool = True,
        min_quality_score: float = 0.5,
        batch_size: int = 20,
    ):
        self.llm = llm
        self.prompt_version = prompt_version
        self.enable_quality_control = enable_quality_control
        self.batch_size = batch_size
        
        self.quality_controller = TripleQualityController(
            enable_fuzzy_dedup=True
        ) if enable_quality_control else None
        
        self.min_quality_score = min_quality_score
    
    async def execute(self, text_blocks: List[TextBlock]):
        """批量处理所有文本块 - 分批API调用"""
        all_triples = []
        successful_batches = 0
        
        if not text_blocks:
            return {
                "all_triples": [],
                "total_blocks": 0,
                "successful_blocks": 0,
            }
        
        # 分批处理
        batches = [
            text_blocks[i : i + self.batch_size]
            for i in range(0, len(text_blocks), self.batch_size)
        ]
        
        print(f"  [*] 将 {len(text_blocks)} 个文本块分为 {len(batches)} 个批次处理 (每批 {self.batch_size} 个)")
        
        for i, batch in enumerate(batches):
            try:
                combined_text = self._combine_blocks(batch)
                system_prompt, user_prompt = self._build_prompts(combined_text)
                
                messages = [
                    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                    ChatMessage(role=MessageRole.USER, content=user_prompt),
                ]
                
                # print(f"    Processing batch {i+1}/{len(batches)}...")
                response = await self.llm.achat(messages)
                content = response.message.content
                
                triples_data = self._parse_llm_response(content)
                
                if triples_data:
                    successful_batches += 1
                    # 为每个三元组添加元数据（使用批次中第一个块的信息作为来源参考）
                    for triple_dict in triples_data:
                        if "source_text_id" not in triple_dict:
                            triple_dict["source_text_id"] = f"batch_{i+1}"
                        if "source_file" not in triple_dict:
                            triple_dict["source_file"] = batch[0].source_file
                        all_triples.append(triple_dict)
                
            except Exception as e:
                print(f"    [!] 批次 {i+1} 处理失败: {str(e)}")
        
        return {
            "all_triples": all_triples,
            "total_blocks": len(text_blocks),
            "successful_blocks": successful_batches * self.batch_size, # 估算值
        }
    
    def _combine_blocks(self, text_blocks: List[TextBlock]) -> str:
        """合并所有文本块为一个文本"""
        combined_parts = []
        for i, block in enumerate(text_blocks):
            combined_parts.append(f"[块{i+1}] {block.text}")
        return "\n\n".join(combined_parts)
    
    def _build_prompts(self, text: str) -> tuple[str, str]:
        """
        构建提示词（调用提示词模块）
        """
        try:
            version_enum = PromptVersion(self.prompt_version)
        except ValueError:
            version_enum = PromptVersion.V3_ACADEMIC
        
        system_prompt = TripleExtractionPrompts.build_system_prompt(version_enum)
        user_prompt = TripleExtractionPrompts.build_user_prompt(
            text=text,
            max_length=2500,
            include_examples=False,
            version=version_enum
        )
        
        return system_prompt, user_prompt
    
    def _parse_llm_response(self, content: str) -> List[Dict[str, Any]]:
        """
        解析 LLM 响应（增强版）
        """
        try:
            # 尝试直接解析
            data = json.loads(content)
            triples = data.get("triples", [])
        except json.JSONDecodeError:
            # 尝试提取 JSON 片段
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = content[start:end + 1]
                data = json.loads(snippet)
                triples = data.get("triples", [])
            else:
                return []
        
        # 🆕 质量控制
        if self.enable_quality_control and self.quality_controller:
            triples = self.quality_controller.deduplicate_triples(
                triples,
                use_global=True
            )
            triples = self.quality_controller.filter_by_quality(
                triples,
                min_score=self.min_quality_score
            )
        
        return triples


# ==================== 数据加载器 ====================

class JSONDocumentLoader:
    """JSON 文档加载器"""
    
    @staticmethod
    def load_from_file(json_path: Path) -> List[TextBlock]:
        """从 JSON 文件加载文本块"""
        doc_json = json.loads(json_path.read_text(encoding="utf-8"))
        blocks = []
        
        # 格式 1：带页面结构
        if "pages" in doc_json:
            for page in doc_json["pages"]:
                page_num = page.get("page_number", "?")
                for idx, block in enumerate(page.get("blocks", [])):
                    text = block.get("text", "").strip()
                    if text and len(text) > 10:
                        blocks.append(TextBlock(
                            text=text,
                            source_text_id=f"p{page_num}-b{idx}",
                            source_file=json_path.name,
                            page_number=page_num if isinstance(page_num, int) else None,
                        ))
        
        # 格式 2：扁平内容
        elif "content" in doc_json:
            for idx, item in enumerate(doc_json["content"]):
                text = item.get("text", "").strip()
                page_num = item.get("page")
                if text and len(text) > 10:
                    blocks.append(TextBlock(
                        text=text,
                        source_text_id=f"p{page_num or '?'}-b{idx}",
                        source_file=json_path.name,
                        page_number=page_num,
                    ))

        # 格式 3：结构化 sections（来自 pdf_to_json.py 输出）
        elif "structure" in doc_json and isinstance(doc_json.get("structure"), dict):
            sections = doc_json["structure"].get("sections", [])
            for s_idx, section in enumerate(sections or []):
                contents = section.get("content", [])
                for c_idx, item in enumerate(contents or []):
                    text = (item.get("text") or "").strip()
                    ctype = item.get("type") or item.get("content_type") or "paragraph"
                    page_num = item.get("page")
                    if not text or len(text) <= 10:
                        continue
                    # 抽取文本内容，忽略图片/表格
                    if ctype in ("text", "paragraph", "list_item"):
                        blocks.append(TextBlock(
                            text=text,
                            source_text_id=f"s{s_idx}-c{c_idx}",
                            source_file=json_path.name,
                            page_number=page_num if isinstance(page_num, int) else None,
                        ))
        
        return blocks


# ==================== 环境配置 ====================

def load_env_file(env_path: Path) -> None:
    """加载 .env 文件"""
    if not env_path.exists():
        return
    
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and value:
                os.environ.setdefault(key, value)


def build_llm(model: Optional[str] = None, temperature: float = 0.1) -> LlamaOpenAI:
    """构建 LLM"""
    base_url = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
    api_key = os.environ.get("YUNWU_API_KEY")
    if not api_key:
        raise RuntimeError("未找到YUNWU_API_KEY，请在 .env 设置")
    
    mdl = model or os.environ.get("YUNWU_MODEL", "gpt-4o-mini")
    
    llm = LlamaOpenAI(
        model=mdl,
        api_key=api_key,
        api_base=base_url,
        temperature=temperature,
        max_tokens=2000,  # 🆕 增加输出长度
    )
    Settings.llm = llm
    return llm


# ==================== 主处理流程 ====================

async def process_single_file(
    json_file: Path,
    output_dir: Path,
    workflow: TripleExtractionWorkflow,
) -> Dict[str, Any]:
    """处理单个 JSON 文件"""
    
    try:
        print(f"\n[*] 正在处理文件: {json_file.name}")
        loader = JSONDocumentLoader()
        text_blocks = loader.load_from_file(json_file)
        
        if not text_blocks:
            print(f"  [!] 跳过: 无有效文本块")
            return {
                "file": json_file.name,
                "status": "skipped",
                "reason": "无有效文本块",
            }
        
        print(f"  [*] 加载了 {len(text_blocks)} 个文本块")
        
        result = await workflow.execute(text_blocks)
        
        # 保存结果
        output_file = output_dir / f"{json_file.stem}.triples.json"
        output_data = {
            "source_file": json_file.name,
            "total_blocks": result["total_blocks"],
            "successful_blocks": result["successful_blocks"],
            "total_triples": len(result["all_triples"]),
            "triples": result["all_triples"],
        }
        
        output_file.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        print(f"  [OK] 抽取了 {len(result['all_triples'])} 个高质量三元组")
        
        return {
            "file": json_file.name,
            "status": "success",
            "output": output_file.name,
            "blocks": result["total_blocks"],
            "successful_blocks": result["successful_blocks"],
            "triples": len(result["all_triples"]),
        }
        
    except Exception as e:
        print(f"  ❌ 处理失败: {str(e)[:200]}")
        return {
            "file": json_file.name,
            "status": "failed",
            "error": str(e),
        }


# ==================== CLI 入口 ====================

@click.command()
@click.option(
    "--input", "input_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="输入目录"
)
@click.option(
    "--output", "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="输出目录"
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="LLM 模型"
)
@click.option(
    "--prompt-version",
    default="v3_academic",
    type=click.Choice(["v3_academic"]),
    help="提示词版本"
)
@click.option(
    "--sample",
    default=10,
    type=int,
    help="抽样文件数"
)
@click.option(
    "--disable-quality-control",
    is_flag=True,
    help="禁用质量控制"
)
@click.option(
    "--min-quality-score",
    default=0.5,
    type=float,
    help="最低质量分数"
)
@click.option(
    "--batch-size",  # 🆕 批次大小
    default=20,
    type=int,
    help="批次大小"
)
def main(
    input_dir: Path,
    output_dir: Path,
    model: str,
    prompt_version: str,
    sample: int,
    disable_quality_control: bool,
    min_quality_score: float,
    batch_size: int,
):
    """
    基于 LlamaIndex 的知识三元组抽取工具（增强版）
    """
    
    print("[*] 启动知识三元组抽取工具（增强版）\n")
    
    # 加载环境
    load_env_file(Path(".env"))
    
    # 配置 LLM
    print("[*] 配置:")
    print(f"  - 模型: {model}")
    print(f"  - 提示词版本: {prompt_version}")
    print(f"  - 质量控制: {'关闭' if disable_quality_control else '开启'}")
    print(f"  - 最低质量分数: {min_quality_score}")
    print(f"  - 批次大小: {batch_size}")
    
    llm = build_llm(model, temperature=0.1)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描文件
    json_files = sorted([
        p for p in input_dir.glob("*.json")
        if p.name != "manifest.json"
    ])
    
    if not json_files:
        print(f"[X] 未找到 JSON 文件: {input_dir}")
        return
    
    if sample > 0 and len(json_files) > sample:
        json_files = random.sample(json_files, sample)
        print(f"[*] 随机抽样 {sample} 个文件\n")
    
    workflow = TripleExtractionWorkflow(
        llm=llm,
        prompt_version=prompt_version,
        enable_quality_control=not disable_quality_control,
        min_quality_score=min_quality_score,
        batch_size=batch_size,
    )
    
    # 处理文件
    async def process_all():
        for json_file in tqdm(json_files, desc="处理文件"):
            await process_single_file(json_file, output_dir, workflow)
    
    asyncio.run(process_all())
    
    print("\n[OK] 所有任务处理完成！")


if __name__ == "__main__":
    main()
