from typing import Any, List, Optional
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _protect_terms(text: str, protect_terms: List[str]) -> str:
        """
        在分割前，将需要保护的术语中的句号替换为特殊标记 [DOT]。
        """
        for term in protect_terms:
            text = text.replace(term, term.replace(".", "[DOT]"))
        return text

def _restore_terms(text: str) -> str:
    """
    在分割后，将特殊标记还原为原始句号。
    """
    return text.replace("[DOT]", ".")

class ChineseSentenceTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s",
        ]
        self._is_separator_regex = is_separator_regex
    
    
    def _split_text(self, text: str, separators: List[str],chunk_size=300, chunk_overlap=50, protect_terms= None) -> List[str]:
    # def split_text(text: str, chunk_size: int, chunk_overlap: int, protect_terms: List[str] = None) -> List[str]:
        """
        分割文本，避免缩写中的句号影响分割，同时保证一定的字符重叠且不分割单词。
        """
        protect_terms = protect_terms or ['Z. mobilis','Z.mobilis', 'z. cervisiae','S.cerevisiae','B.subtilis',
                       'E. coli','B. licheniformis','B. Evolutionary','S.stipnis','Z.mobilis','CVtree3.0','K.pneumoniae','K.oxyroca','B. subtilis']

        # 保护术语
        text = _protect_terms(text, protect_terms)

        # 按句子拆分
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                # 如果当前块达到长度限制，则保存并创建新块
                chunks.append(current_chunk.strip())

                # 保证重叠，调整切分点不拆单词
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                overlap_part = current_chunk[overlap_start:]
                if " " in overlap_part:
                    overlap_start = current_chunk.rfind(" ", 0, overlap_start) + 1

                current_chunk = current_chunk[overlap_start:].strip() + " " + sentence + " "

        # 添加最后的块
        if current_chunk:
            chunks.append(current_chunk.strip())
        print("+++++++ChineseSentenceTextSplitter+++++++")
        # 还原术语
        # return chunks
        return [_restore_terms(chunk) for chunk in chunks]

