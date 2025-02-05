import openai  # Imports the OpenAI library for interacting with the API
from openai import OpenAI  # Imports OpenAI client class (though not directly needed)

class grade_doc:
    """
    A class to grade the relevance of a retrieved document to a user question using the OpenAI API.

    Attributes:
        openai_api_key (str): The API key for authenticating with the OpenAI API.
        model (str): The name of the OpenAI model to use for grading (default is 'gpt-4').
        grade_msg (str): Instruction for the GPT model to assess relevance and provide a binary score.
    """

    def __init__(self, openai_api_key: str, model="gpt-4"):
        """
        Initializes the grade_doc class with an API key and model name.

        Args:
            openai_api_key (str): The API key for OpenAI API.
            model (str): The name of the OpenAI model to use (default is 'gpt-4').

        Sets:
            self.api_key: Stores the API key for further use.
            self.client: Initializes the OpenAI API client using the provided API key.
            self.model: Stores the name of the model to be used for grading.
            self.grade_msg: A fixed instruction message to guide the model on how to assess relevance.
        """
        openai.api_key = openai_api_key  # Sets the API key for the openai module.
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=self.api_key)  # Creates an OpenAI client object.
        self.model = model
        self.grade_msg = f"""You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    def grade_document(self, query, context):
        """
        Grades the relevance of a document to a given user query.

        Args:
            query (str): The user's question or query.
            context (str): The retrieved document to be graded for relevance.

        Returns:
            str: A binary score ('yes' or 'no') indicating if the document is relevant to the query.
        """

        # Creates a prompt with the user query and document context.
        prompt = f"""
        User Question : {query}
        Document : {context}"""

        # Sends the prompt and grading instruction to the OpenAI API and gets a response.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.grade_msg},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extracts the result from the API response and returns the binary score ('yes' or 'no').
        score = response.choices[0].message.content.strip().lower()
        return score
