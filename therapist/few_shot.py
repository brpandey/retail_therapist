from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.prompts import load_prompt

PROMPT_PATH = "./therapist/cs_few_shot.json"


def create_few_shot_prompt():
    # . Create few-shot examples
    examples = [
        {
            "customer_complaint": "Wow, this place is a joke. They wouldn't even tape my box shut. I guess customer service doesn't exist here.",
            "response": "We're really sorry to hear your visit left that impression. We used to tape boxes as a courtesy, but over time, many packages began arriving completely unsealed, and unfortunately, some folks started relying on us for full repackaging—which became tough to sustain. We now kindly ask that shipments come pre-sealed, but we understand how frustrating a change in expectations can be. We'll take this as a reminder to keep finding ways to serve more clearly and kindly.",
        },
        {
            "customer_complaint": "Every other store helps with this stuff. Why is this one so difficult?! I don't have time to argue over a roll of tape.",
            "response": "We hear you—it's never our intention to add stress to your day. While some locations may still provide taping assistance, our store had to update our policy after seeing a large increase in customers needing full repackaging help. It became unsustainable, both in cost and time. We totally get that when you're in a hurry, this feels like one more hassle. We'll keep working on how we communicate these policies clearly and kindly. Your time matters to us.",
        },
    ]

    # . Define the example template
    example_template = """
        CUSTOMER REVIEW:
        {customer_complaint}

        YOUR RESPONSE:
        {response}
    """

    example_prompt = PromptTemplate(
        input_variables=["customer_complaint", "response"], template=example_template
    )

    # Create the RetrievalQA chain
    #    qa_chain = RetrievalQA.from_chain_type(
    #        llm=ChatOpenAI(), chain_type="stuff", retriever=retriever
    #    )

    # Ask a question
    #     response = qa_chain.invoke(
    #         {"query": "What's the store's policy on packaging tape?"}
    #     )

    # Creates a custom prompt template that frames the task as responding to customer reviews while considering store policy

    # Use the following store policy context to craft a helpful, understanding response to the customer review.

    prefix_template = """You are a customer service representative responding to online reviews.
        Please provide a response to the customer feedback with three different styles using the following store policy.

        One style is a casual, fun, cheerful, with the air of a very refined and articulate woman.
        Another style is humorous, relatable and adventurous.
        The third style is jovial and practical, like a loyal trusty neighbor.

        Gently educate the customer about the policy rationale.

        Avoid platitudes. Have a real simple conversation, and use natural language.
        Please provide three different replies given the three styles to above following customer
        review feedback

        Please also inject a few details from the original customer review so it feels authentic. 
        Please make response more than 6 sentences long in length, but no need for emoticons!

        STORE POLICY CONTEXT:
        {context}

        By the way, if you are not fully prepared to answer based on the customer feedback and the context please reply
        with a conciliatory message that you don't know.

        Here are some examples of good responses to customer complaints:"""

    suffix_template = """
        CUSTOMER REVIEW:
        {question}

        {format_instructions}

        YOUR RESPONSE:"""

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix_template,
        suffix=suffix_template,
        input_variables=["context", "question", "format_instructions"],
    )

    few_shot_prompt.save(PROMPT_PATH)


def load_few_shot_prompt(output_parser):
    format_instructions = output_parser.get_format_instructions()

    few_shot = load_prompt(PROMPT_PATH)
    result = few_shot.partial(format_instructions=format_instructions)

    return result


def create_output_parser():
    # Define the output schemas for three different response styles
    response_schemas = [
        ResponseSchema(
            name="empathetic_response",
            description="A warm, understanding response that prioritizes the customer's feelings and acknowledges their frustration",
        ),
        ResponseSchema(
            name="educational_response",
            description="A response that clearly explains the policy rationale and educates the customer in a friendly way",
        ),
        ResponseSchema(
            name="solution_oriented_response",
            description="A response focused on practical solutions and alternatives for the customer's situation",
        ),
    ]

    # Create the output parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser
