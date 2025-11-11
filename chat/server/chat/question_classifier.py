from typing import Literal, Optional
import re


QuestionType = Literal[
    "simple_qa",
    "multiple_choice",
    "true_false",
    "default"
]


class QuestionClassifier:
    def __init__(self, use_llm: bool = True, llm_model: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_model = llm_model
    
    def classify_by_rules(self, question: str) -> Optional[QuestionType]:
        question_lower = question.lower().strip()
        
        choice_patterns = [
            r'\b[A-D][\.:\)]',
            r'\b(option|choice)\s+[A-D]',
            r'which\s+(of\s+the\s+following|one|statement)',
            r'select\s+(the|all|one)',
        ]
        for pattern in choice_patterns:
            if re.search(pattern, question_lower):
                return "multiple_choice"
        
        true_false_patterns = [
            r'^(true\s+or\s+false|t/f)',
            r'(is\s+this|is\s+it|is\s+the\s+statement)\s+(true|false|correct)',
            r'(determine|judge|decide)\s+(whether|if)',
        ]
        for pattern in true_false_patterns:
            if re.search(pattern, question_lower):
                return "true_false"
        return None
    
    async def classify_by_llm(self, question: str) -> QuestionType:
        try:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            
            classification_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a question type classification expert. Classify the user's question into one of the following types:

1. simple_qa - Simple Q&A (general questions like "What is...", "How to...")
2. multiple_choice - Multiple choice questions (questions with options A/B/C/D)
3. true_false - True/False questions (judge correctness, true or false)
4. default - Other types

Only respond with the type name, no other content.

Note: Questions may be in English or Chinese. Analyze accordingly."""),
                ("human", "Question: {question}")
            ])
            
            llm = ChatOpenAI(model=self.llm_model, temperature=0)
            chain = classification_prompt | llm
            result = await chain.ainvoke({"question": question})
            
            classification = result.content.strip().lower()
            
            valid_types = [
                "simple_qa", "multiple_choice", "true_false", "default"
            ]
            
            for valid_type in valid_types:
                if valid_type in classification:
                    return valid_type
            
            return "default"
        except Exception as e:
            return "default"
    
    async def classify(self, question: str) -> QuestionType:
        if not self.use_llm:
            return "simple_qa"
        
        rule_result = self.classify_by_rules(question)
        if rule_result:
            return rule_result
        
        try:
            return await self.classify_by_llm(question)
        except Exception as e:
            return "simple_qa"
    
    def get_prompt_name(self, question_type: QuestionType) -> str:
        return question_type


_classifier_instance: Optional[QuestionClassifier] = None


def get_classifier(use_llm: bool = True, llm_model: str = "gpt-4o-mini") -> QuestionClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QuestionClassifier(use_llm=use_llm, llm_model=llm_model)
    return _classifier_instance

