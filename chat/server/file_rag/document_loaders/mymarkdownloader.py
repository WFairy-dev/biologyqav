from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from typing import List
import re,json


class RapidOCRMarkdownLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def process_markdown(file_path):
            # current_directory = os.getcwd()
            # source=os.path.join(current_directory, file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            
            # 去除不需要的内容
            md_content = re.sub(r'[*>]', '', md_content)
            
            lines = md_content.split('\n')
            processed_lines = []

            for line in lines:
                if line.startswith('#'):
                    title = line.lstrip('#').strip() + '\n'
                    processed_lines.append(title)
                else:
                    processed_lines.append(line)
            
            processed_content = '\n'.join(processed_lines)
            return processed_content
        
        text = process_markdown(self.file_path)
        from unstructured.partition.text import partition_text
        print("=============使用markdownloader进行文件加载================")
        return partition_text(text=text, **self.unstructured_kwargs)

