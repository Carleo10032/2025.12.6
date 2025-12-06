import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import re

import click
import fitz


class ParagraphParser:
    """段落解析器"""
    
    def __init__(self):
        self.current_section = None
        self.section_counter = 0
        
    def identify_section_type(self, text: str, font_size: float, bbox: tuple) -> str:
        """识别段落类型"""
        text_lower = text.lower().strip()
        
        patterns = [
            (lambda: any(text_lower.startswith(s) for s in 
                ['abstract', 'introduction', 'methods', 'methodology', 
                 'results', 'discussion', 'conclusion', 'references']), "section_title"),
            (lambda: font_size > 13 and len(text.split()) < 15, "subsection_title"),
            (lambda: re.search(r'\d{1,2}(Department|College|University|Institute)', text), "author_affiliation"),
            (lambda: text_lower.startswith('keywords'), "keywords"),
            (lambda: re.match(r'^(Fig\.|Figure|Table)\s+\d+', text, re.IGNORECASE), "figure_caption"),
            (lambda: re.search(r'[=∫∑∏]', text) and len(text) < 200, "formula"),
            (lambda: re.match(r'^(\d+\.|[•\-])\s+', text), "list_item"),
            (lambda: re.match(r'^\d+\.\s+\w+,\s+\w+', text), "reference"),
        ]
        
        for condition, type_name in patterns:
            if condition():
                return type_name
        
        return "paragraph"


class DocumentStructure:
    """文档结构管理"""
    
    def __init__(self):
        self.sections = []
        self.current_section = None
        self.paragraph_parser = ParagraphParser()
        
    def add_section(self, title: str, level: int, page: int):
        """添加新章节"""
        section = {
            "id": f"section_{len(self.sections)}",
            "title": title,
            "level": level,
            "page": page,
            "content": []
        }
        self.sections.append(section)
        self.current_section = section
        return section
    
    def add_content(self, content_type: str, text: str, page: int, 
                   font_size: float, bbox: tuple, metadata: Dict = None):
        """添加内容到当前章节"""
        if not self.current_section:
            # 如果还没有章节，创建一个默认章节
            self.add_section("Document Content", 1, page)
        
        content_item = {
            "type": content_type,
            "text": text,
            "page": page,
            "font_size": font_size,
            "bbox": bbox
        }
        
        if metadata:
            content_item.update(metadata)
        
        self.current_section["content"].append(content_item)


def merge_continuous_text(blocks: List[Dict]) -> List[Dict]:
    """合并连续的文本块形成段落"""
    if not blocks:
        return []
    
    merged = []
    current = None
    non_mergeable = ["image", "table", "section_title", "subsection_title", 
                     "figure_caption", "formula", "keywords"]
    
    def should_merge(curr, block):
        if block.get("type") in ["image", "table"]:
            return False
        block_type = block.get("content_type", "paragraph")
        if block_type in non_mergeable:
            return False
        if not curr:
            return False
        return (block["page"] == curr["page"] and
                abs(block["font_size"] - curr["font_size"]) < 0.5 and
                block["bbox"][1] - curr["bbox"][3] < 15 and
                block_type == curr.get("content_type", "paragraph"))
    
    for block in blocks:
        if not should_merge(current, block):
            if current:
                merged.append(current)
            current = block.copy()
        else:
            current["text"] += " " + block["text"]
            current["bbox"] = [
                min(current["bbox"][0], block["bbox"][0]),
                min(current["bbox"][1], block["bbox"][1]),
                max(current["bbox"][2], block["bbox"][2]),
                max(current["bbox"][3], block["bbox"][3])
            ]
    
    if current:
        merged.append(current)
    
    return merged


def extract_tables(page, page_num: int) -> List[Dict]:
    """提取表格"""
    try:
        return [{
            "type": "table",
            "page": page_num,
            "table_id": f"table_{page_num}_{i}",
            "bbox": table.bbox,
            "rows": [[cell or "" for cell in row] for row in table.extract()]
        } for i, table in enumerate(page.find_tables())]
    except:
        return []


def clean_text(text: str) -> str:
    """清理文本"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()


def extract_pdf_with_paragraphs(pdf_path: Path) -> Dict[str, Any]:
    """分段解析PDF内容"""
    doc = fitz.open(pdf_path)
    doc_structure = DocumentStructure()
    parser = ParagraphParser()
    
    all_blocks = []
    all_tables = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        all_tables.extend(extract_tables(page, page_num + 1))
        
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") == 0:
                font_sizes = []
                lines = []
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    font_sizes.extend(span.get("size", 0) for span in line.get("spans", []))
                    if line_text.strip():
                        lines.append(line_text.strip())
                
                text = clean_text(" ".join(lines)) if lines else ""
                if not text:
                    continue
                
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                bbox = block.get("bbox")
                
                all_blocks.append({
                    "page": page_num + 1,
                    "text": text,
                    "bbox": list(bbox),
                    "font_size": round(avg_font_size, 1),
                    "content_type": parser.identify_section_type(text, avg_font_size, bbox)
                })
            
            elif block.get("type") == 1:
                all_blocks.append({
                    "page": page_num + 1,
                    "type": "image",
                    "bbox": list(block.get("bbox")),
                    "width": block.get("width"),
                    "height": block.get("height"),
                    "content_type": "image"
                })
    
    page_count = doc.page_count
    doc.close()
    
    # 合并连续文本形成段落
    merged_blocks = merge_continuous_text(all_blocks)
    
    for block in merged_blocks:
        content_type = block.get("content_type", "paragraph")
        
        if content_type in ["section_title", "subsection_title"]:
            doc_structure.add_section(
                title=block["text"],
                level=1 if content_type == "section_title" else 2,
                page=block["page"]
            )
        else:
            doc_structure.add_content(
                content_type=content_type,
                text=block.get("text", ""),
                page=block["page"],
                font_size=block.get("font_size", 0),
                bbox=block["bbox"],
                metadata={"type": block.get("type", "text")}
            )
    
    # 构建最终结构
    return {
        "metadata": {
            "filename": pdf_path.name,
            "source": str(pdf_path),
            "pages": page_count,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "parser_version": "3.0_paragraph"
        },
        "structure": {
            "sections": doc_structure.sections
        },
        "tables": all_tables,
        "statistics": {
            "total_sections": len(doc_structure.sections),
            "total_paragraphs": sum(
                len([c for c in s["content"] if c["type"] == "paragraph"]) 
                for s in doc_structure.sections
            ),
            "total_tables": len(all_tables),
            "total_images": sum(
                len([c for c in s["content"] if c.get("type") == "image"]) 
                for s in doc_structure.sections
            )
        }
    }


@click.command()
@click.option("--input", "input_dir", default="./agent/datas", 
              type=click.Path(path_type=Path), help="PDF文件所在目录")
@click.option("--output", "output_dir", default="./output", 
              type=click.Path(path_type=Path), help="JSON输出目录")
@click.option("--overwrite", is_flag=True, help="是否覆盖已存在的文件")
@click.option("--format", "output_format", 
              type=click.Choice(['structured', 'flat']), 
              default='structured',
              help="输出格式:structured(分段结构) 或 flat(扁平列表)")
def main(input_dir: Path, output_dir: Path, overwrite: bool, output_format: str):
    """将PDF文件批量解析为JSON格式"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdfs = list(input_dir.glob("*.pdf"))
    if not pdfs:
        click.echo(f"❌ 在 {input_dir} 未找到PDF文件")
        return
    
    click.echo(f"📁 找到 {len(pdfs)} 个PDF文件")
    click.echo(f"📊 输出格式: {output_format}\n")
    
    results = []
    for pdf_path in pdfs:
        output_file = output_dir / f"{pdf_path.stem}.json"
        
        if output_file.exists() and not overwrite:
            click.echo(f"⊘ {pdf_path.name} (已存在)")
            results.append({"file": pdf_path.name, "status": "skipped"})
            continue
        
        try:
            data = extract_pdf_with_paragraphs(pdf_path)
            
            output_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            stats = data["statistics"]
            click.echo(
                f"✓ {pdf_path.name}\n"
                f"  └─ {stats['total_sections']} 章节, "
                f"{stats['total_paragraphs']} 段落, "
                f"{stats['total_tables']} 表格, "
                f"{stats['total_images']} 图片"
            )
            results.append({"file": pdf_path.name, "status": "success"})
            
        except Exception as e:
            click.echo(f"✗ {pdf_path.name} - {str(e)}")
            import traceback
            click.echo(traceback.format_exc())
            results.append({"file": pdf_path.name, "status": "error", "error": str(e)})
    
    # 生成处理报告
    success = sum(1 for r in results if r["status"] == "success")
    click.echo(f"\n{'='*50}")
    click.echo(f"✅ 完成: {success}/{len(pdfs)} 个文件处理成功")
    
    # 保存处理报告
    report_file = output_dir / "processing_report.json"
    report_file.write_text(
        json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_files": len(pdfs),
            "successful": success,
            "results": results
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    click.echo(f"📄 处理报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
