from enum import Enum


class PromptVersion(Enum):
    V3_ACADEMIC = "v3_academic"


class TripleExtractionPrompts:
    @staticmethod
    def build_system_prompt(version: PromptVersion) -> str:
        if version == PromptVersion.V3_ACADEMIC:
            return (
                "你是学术信息抽取助手。只返回JSON,键为 triples。请提取文本中核心的学术知识三元组(因果、定义、属性、方法)。"
                "要求:1. 实体和关系必须精炼简洁,subject和object尽量为名词短语,predicate为动词短语。"
                "2. 避免提取冗长、无关紧要或修饰性过强的三元组。"
                "3. 实体名称在保持准确的前提下尽量简短。"
                "4. 不要推断;严格JSON,无任何附加文本。"
            )

    @staticmethod
    def build_user_prompt(
        text: str,
        max_length: int = 2500,
        include_examples: bool = False,
        version: PromptVersion = PromptVersion.V3_ACADEMIC,
    ) -> str:
        t = (text or "")[:max_length]
        examples = ""
        if include_examples:
            examples = (
                "示例:{\"triples\":[{\"subject\":\"锂金属电池\",\"predicate\":\"具有\",\"object\":\"高能量密度\"}]}\n"
            )
        return (
            examples
            + "从以下文本中提取最核心、最精炼的知识三元组,忽略次要信息。仅返回JSON:\n" + t
        )
