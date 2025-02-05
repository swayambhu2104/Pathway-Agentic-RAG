import openai
from openai import OpenAI  



class ConversationalPipeline:

    def __init__(self, openai_api_key, model="gpt-4o"):
        """
        Initializes the ConversationalPipeline with an API key and model.

        Args:
            openai_api_key (str): The API key for OpenAI API.
            model (str): The OpenAI model to use (default: "gpt-4o").
        """
        openai.api_key = openai_api_key  # Sets the global API key
        self.api_key = openai_api_key  # Stores the API key for the client
        self.model = model  # Stores the model name
        self.client = OpenAI(api_key=self.api_key)  # (Redundant usage)



    def call_openai(self, prompt, model="gpt-4"):
        """
        Sends a prompt to the OpenAI API and returns the response.

        Args:
            prompt (str): The prompt to send to the model.
            model (str): The OpenAI model to use (default: "gpt-4").

        Returns:
            str: The response from the API.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}],
            model=self.model,
        )
        return response.choices[0].message.content

    def analyst_task(self, query, context):
        """
        Analyzes a query using the provided document context.

        Args:
            query (str): The user query.
            context (str): The document context.

        Returns:
            str: The analysis and response from the analyst.
        """
        analyst_prompt = f"""
        You are a financial agent. Your task is to analyze the following query using the provided document context:

        Query: {query}
        Document Context: {context}

        Provide an informative response based on the query. If further clarification is needed, suggest a follow-up question. If no follow-up is needed, provide a conclusion.
        """
        return self.call_openai(analyst_prompt)


    def leader_task(self, response_1, response_2, query, context):
        """
        Unifies and summarizes responses from multiple analysts.

        Args:
            response_1 (str): Response from Analyst 1.
            response_2 (str): Response from Analyst 2.
            query (str): The original query.
            context (str): The combined context.

        Returns:
            str: The unified response.
        """
        leader_prompt = f"""
        You are the Leader. Your task is to unify and summarize the responses from Analyst 1 and Analyst 2 into a coherent final response, given the query and the context:

        Query: {query}
        Context: {context}

        Analyst 1 Response: {response_1}
        Analyst 2 Response: {response_2}

        Provide the unified response below.
        """
        return self.call_openai(leader_prompt)

    
    def check_follow_up(self, query, context, final_response):
        """
        Determines if a follow-up is needed for the query.

        Args:
            query (str): The user query.
            context (str): The context used.
            final_response (str): The final response generated.

        Returns:
            str: "Yes" if a follow-up is needed, otherwise "No".
        """
        follow_up_prompt = f"""
        Based on the provided query, context, and final response, determine if the query has been fully answered.

        Query: {query}
        Context: {context}
        Final Response: {final_response}

        Output "Yes" if the query is fully answered, otherwise output "No."
        """
        return self.call_openai(follow_up_prompt).strip()

    
    def divide_correct_task_into_subtasks(self, query, context):
        """
        Divides a query into two distinct subtasks based on the provided context if the query has been classified as correct, i.e, 
        it can be solved using the context provided.

        Args:
            query (str): The user query.
            context (str): The document context.

        Returns:
            tuple: Two distinct subtasks (subtask_1, subtask_2).
        """
        divide_prompt = f"""
        The user has a query and a context. Your first task is to determine whether the query is a simple query or complex one i.e., determine whether there is a need of dividing the query into simple ones or not.
        If the query is simple, then keep the subtask1 as initial query and subtask 2 as empty, else your next task is to divide this query into two distinct subtasks that can be worked on independently.
        Subtask 1: Focuses on one fundamental aspect of the query.
        Subtask 2: Focuses on a complementary or orthogonal aspect to Subtask 1, without overlapping in scope or objectives.
        Ensure that the subtasks are distinct, independent, and non-redundant.
        Instructions:
        Analyze the provided query and context.
        Identify two subtasks that:
        Together contribute to solving the overall query.
        Individually focus on different aspects of the problem to avoid redundancy.
        Ensure the subtasks are well-defined and actionable, with clear objectives.
        Here is the query and context:
        Query: "{query}"
        Context: "{context}"

        Provide the output in this format:

        Subtask 1: [In case of simple query, keep the initial query here else in case of complex query keep the first independent subtask with clear and actionable instructions]
        Subtask 2: [In case of simple query, keep this empty else in case of complex query keep the second distinct subtask that complements the first]
        """
        response = self.call_openai(divide_prompt)
        subtask_1, subtask_2 = response.split("Subtask 1:")[1].split("Subtask 2:")
        return subtask_1.strip(), subtask_2.strip()


    def divide_incorrect_task_into_subtasks(self, query):

        """
        Divides a query into two distinct subtasks based on the provided context if the query has been classified as incorrect, i.e, it cannot 
        be solved using the context provided and needs additional web search.

        Args:
            query (str): The user query.
            context (str): The document context.

        Returns:
            tuple: Two distinct subtasks (subtask_1, subtask_2).
        """

        divide_prompt = f"""
        The user has a query. Your first task is to determine whether the query is a simple query or complex one i.e., determine whether there is a need of dividing the query into simple ones or not.
        If the query is simple, then keep the subtask1 as initial query and subtask 2 as empty, else your next task is to divide this query into two distinct subtasks that can be worked on independently.
        Subtask 1: Focuses on one fundamental aspect of the query.
        Subtask 2: Focuses on a complementary or orthogonal aspect to Subtask 1, without overlapping in scope or objectives.
        Ensure that the subtasks are distinct, independent, and non-redundant.
        Instructions:
        Analyze the provided query.
        Identify two subtasks that:
        Together contribute to solving the overall query.
        Individually focus on different aspects of the problem to avoid redundancy.
        Ensure the subtasks are well-defined and actionable, with clear objectives.
        Here is the query:
        Query: "{query}"

        Provide the output in this format:

        Subtask 1: [In case of simple query, keep the initial query here else in case of complex query keep the first independent subtask with clear and actionable instructions]
        Subtask 2: [In case of simple query, keep this empty else in case of complex query keep the second distinct subtask that complements the first]
        """
        response = self.call_openai(divide_prompt)
        subtask_1, subtask_2 = response.split("Subtask 1:")[1].split("Subtask 2:")
        return subtask_1.strip(), subtask_2.strip()

    def generate_new_subtasks(self, query, subtask_1, subtask_2, context):
        """
        Generates two new subtasks (Subtask 3 and Subtask 4) based on a query, context, 
        and two existing subtasks. The new subtasks address unexplored aspects of the query.

        Args:
            query (str): The original query provided by the user.
            subtask_1 (str): The first subtask derived from the query.
            subtask_2 (str): The second subtask derived from the query.
            context (str): Additional information or document context related to the query.

        Returns:
            tuple: A tuple containing two strings - Subtask 3 and Subtask 4, both distinct 
                from the first two subtasks and each other.
        """

        # Constructs a prompt for the OpenAI model to generate new subtasks.
        generate_new_subtasks_prompt = f"""
        You are provided with a query, its context, and two previously defined subtasks (Subtask 1 and Subtask 2). Your task is to generate two new subtasks: Subtask 3 and Subtask 4, ensuring they are:

        Distinct from both Subtask 1 and Subtask 2.
        Independent of each other but contribute toward solving the original query.
        Complementary to the overall solution while addressing unexplored aspects of the query.

        Instructions:
        1. Analyze the provided query and context carefully.
        2. Avoid reusing or overlapping with the objectives of Subtask 1 and Subtask 2.
        3. Identify two new subtasks (Subtask 3 and Subtask 4) that focus on unexplored dimensions, technical considerations, or potential enhancements.
        4. Make each subtask clear, actionable, and non-redundant with previous subtasks.

        Here is the query, context, and previous subtasks:
        Query: "{query}"
        Context: "{context}"
        Subtask 1: {subtask_1}
        Subtask 2: {subtask_2}

        Provide the output in this format:

        Subtask 3: [Third independent subtask, distinct from Subtask 1 and Subtask 2, focusing on a new actionable aspect of the query]
        Subtask 4: [Fourth independent subtask, distinct from Subtask 1, Subtask 2, and Subtask 3, complementing the overall solution]
        """

        # Calls the OpenAI API with the generated prompt.
        response = self.call_openai(generate_new_subtasks_prompt)

        # Splits the response from the API into Subtask 3 and Subtask 4 based on the prompt format.
        subtask_3, subtask_4 = response.split("Subtask 3:")[1].split("Subtask 4:")

        # Returns the new subtasks after stripping any extra whitespace.
        return subtask_3.strip(), subtask_4.strip()


    def run_pipeline(self, query, context_a, context_b, subtask_1, subtask_2):
        """
        Orchestrates the pipeline for generating responses from two Analyst tasks for distinct subtasks 
        and consolidates them into a final response using a Leader task.

        Args:
            query (str): The original query to be addressed.
            context_a (str): Contextual information for Subtask 1.
            context_b (str): Contextual information for Subtask 2.
            subtask_1 (str): The first subtask derived from the query.
            subtask_2 (str): The second subtask derived from the query.

        Returns:
            str: A final unified response from the Leader after analyzing both Analyst responses.
        """

        # Generate response for Subtask 1 using the context_a document.
        response_1 = self.analyst_task(subtask_1, context_a)

        # Generate response for Subtask 2 using the context_b document.
        response_2 = self.analyst_task(subtask_2, context_b)

        # Consolidate the responses from Analyst 1 and Analyst 2 using the Leader task.
        final_response = self.leader_task(response_1, response_2, query, context_a + context_b)

        # Return the unified final response.
        return final_response


    def final_unification_task(self, combined_response, response_3, response_4, query, context):
        """
        Unifies the responses from multiple Analysts into a single, comprehensive final response 
        to address the original user query.

        Args:
            combined_response (str): Unified response from Analyst 1 and Analyst 2.
            response_3 (str): Response from Analyst 3.
            response_4 (str): Response from Analyst 4.
            query (str): The original query from the user.
            context (str): The overall context or background information for the query.

        Returns:
            str: A final, unified response that integrates inputs from all analysts.
        """

        # Prompt template for the Leader to consolidate all responses into one final response.
        unification_prompt = f"""
        You are the Leader. Your task is to unify and summarize all three analyst responses into a single, coherent, and comprehensive final response, given the query and the context:

        query: {query}
        context: {context}

        Analyst 1_2 Combined Response: {combined_response}
        Analyst 3 Response: {response_3}
        Analyst 4 Response: {response_4}

        Provide the unified response below, ensuring clarity, accuracy, and coherence.
        """

        # Call the OpenAI API to generate the unified response.
        return self.call_openai(unification_prompt)


    def run_pipeline_if_needed(self, query, context_c, context_d, subtask_3, subtask_4, final_response, context):
        """
        Executes additional pipeline tasks if required, handling Subtask 3 and Subtask 4,
        and unifying their responses with the initial final response.

        Args:
            query (str): The original query posed by the user.
            context_c (list or str): Context/data for Subtask 3.
            context_d (list or str): Context/data for Subtask 4.
            subtask_3 (str): The third subtask derived from the original query.
            subtask_4 (str): The fourth subtask derived from the original query.
            final_response (str): The initial combined response from Subtask 1 and Subtask 2.
            context (list): The overall context that will be extended with context_c and context_d.

        Returns:
            str: The unified final response after considering all subtasks.
        """

        # Generate a response for Subtask 3 using the provided context.
        response_3 = self.analyst_task(subtask_3, context_c)
        
        # Generate a response for Subtask 4 using the provided context.
        response_4 = self.analyst_task(subtask_4, context_d)

        # Extend the overall context with the new contexts from Subtask 3 and Subtask 4.
        context.extend(context_c)
        context.extend(context_d)

        # Unify all responses (final_response, response_3, and response_4) into a single, coherent final response.
        final_follow_up_response = self.final_unification_task(final_response, response_3, response_4, query, context)

        # Return the final unified response.
        return final_follow_up_response
