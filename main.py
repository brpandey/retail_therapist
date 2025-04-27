import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.globals import set_debug
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import reviews

# Load environment variables from .env
load_dotenv()
set_debug(False)

# DOC_PATH = "./policies/store.pdf"
EMBEDDING_MODEL = "text-embedding-3-small"


def main():
    st.title("Retail Therapist")

    # Set your OpenAI API key
    # Either get from environment variable or let user input it
    if not os.environ.get("OPENAI_API_KEY", ""):
        os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
            "Enter your OpenAI API key:", type="password"
        )

        if not os.environ.get("OPENAI_API_KEY", ""):
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()

    # Initialize session state to store our retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    st.header("Step 1: Upload Policy Document")
    uploaded_file = st.file_uploader("Upload your policy PDF", type="pdf")

    if uploaded_file is not None:
        # Display success message
        st.success("File uploaded successfully!")

        # Process the PDF when user clicks the button
        if st.button("Process Policy Document"):
            with st.spinner("Processing your PDF..."):
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                    try:
                        st.session_state.retriever = create_retriever(tmp_file_path)
                        st.success("Successfully processed and ingested pdf")
                        os.unlink(tmp_file_path)

                    except Exception as e:
                        # Clean up the temporary file
                        os.unlink(tmp_file_path)

                        st.error(f"Error processing PDF: {e}")
                        st.session_state.retriever = None
                        st.stop()

    output_parser = create_output_parser()

    reviews.display_review_management(st, None)

    if st.session_state.retriever is None:
        st.info(
            "Please upload and process a policy document first, then submit a review."
        )
    elif st.session_state.customer_review is None:
        st.info("Please enter a customer review to get started.")

    if st.button("Generate Response", type="primary"):
        with st.spinner("Generating response..."):
            try:
                # Initialize language model
                model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.7,  # More personality and nuance
                    request_timeout=60,  # Longer timeout
                    model_kwargs={"user": f"user_{int(time.time())}"},  # Unique user ID
                )

                # Load the vector database
                # vector_db = load_vector_db()
                # if vector_db is None:
                #     st.error("Failed to load or create the vector database.")
                #     return

                # Create the prompt
                few_shot_prompt = create_few_shot_prompt(output_parser)

                print("About to invoke chain with feedback")

                # Uses the RetrievalQA chain to generate a thoughtful response to the negative review

                qa_chain = RetrievalQA.from_chain_type(
                    llm=model,
                    chain_type="stuff",
                    retriever=st.session_state.retriever,
                    chain_type_kwargs={"prompt": few_shot_prompt},
                    return_source_documents=True,  # This is important to get the retrieved documents
                )

                # print("Chain is {0}".format(qa_chain))
                print(
                    f"Generate response, customer review is {st.session_state.customer_review}"
                )
                response = qa_chain.invoke({"query": st.session_state.customer_review})

                response = qa_chain.invoke(
                    {"query": f"{st.session_state.customer_review} [ts:{time.time()}]"}
                )
                raw_response = response["result"]

                # Get the retrieved documents
                retrieved_docs = response["source_documents"]

                try:
                    parsed_response = output_parser.parse(raw_response)

                    # Let users choose how many documents to view
                    # max_docs = min(
                    #     len(retrieved_docs), 10
                    # )  # Cap at 10 to avoid UI clutter
                    # num_docs_to_show = st.slider(
                    #     "Number of documents to display", 1, max_docs, 3
                    # )

                    # Display retrieved documents first
                    with st.expander("View Retrieved Context "):
                        st.header("Retrieved Context (Top Documents)")

                        for i, doc in enumerate(retrieved_docs[:6]):
                            st.write(f":blue[Document {i + 1}]")
                            st.write(doc.page_content)
                            # st.write("**Metadata:**")
                            # st.json(doc.metadata)

                    # Create tabs for each response style
                    st.header("Retail Therapist Responses")

                    tab1, tab2, tab3 = st.tabs(
                        ["Empathetic", "Light-hearted", "Neighborly"]
                    )

                    with tab1:
                        st.subheader("Refined Response")
                        st.write(parsed_response["empathetic_response"])

                    with tab2:
                        st.subheader("Light-hearted Response")
                        st.write(parsed_response["educational_response"])

                    with tab3:
                        st.subheader("Practical Response")
                        st.write(parsed_response["solution_oriented_response"])

                    # Show raw JSON for reference
                    # with st.expander("View JSON Response"):
                    #   st.json(parsed_response)

                except Exception as e:
                    st.error(f"Error parsing response: {e}")
                    st.text(raw_response)

                print(response["result"])
                print(st.session_state.retriever)

                # st.markdown("**Assistant:**")
                # st.write(response)

                # Add some instructions
                st.sidebar.header("Instructions")
                st.sidebar.write("""
                1. Upload your policy PDF document
                2. Click 'Process Policy Document'
                3. Enter a customer review or alternatively re-run a saved review
                4. Click 'Generate Responses'
                5. View the different response styles
                """)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


@st.cache_data
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

    # 4. Create the output parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser


def create_retriever(doc_path):
    # 1. Load documents
    loader = PyPDFLoader(file_path=doc_path)
    raw_documents = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    documents = text_splitter.split_documents(raw_documents)

    # 3. Create an embedding function
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,  # Or another embedding model
    )

    # 4. Create a vector store
    # Builds a vector store with this policy information
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embedding_function
    )

    # 5. Create the retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}  # Return top 6 most relevant chunks
    )

    return retriever


def create_few_shot_prompt(output_parser):
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

    format_instructions = output_parser.get_format_instructions()

    # 6. Create the RetrievalQA chain
    #    qa_chain = RetrievalQA.from_chain_type(
    #        llm=ChatOpenAI(), chain_type="stuff", retriever=retriever
    #    )

    # 7. Ask a question
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
        input_variables=["context", "question"],
        partial_variables={"format_instructions": format_instructions},
    )

    return few_shot_prompt

    # prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    # The store's policy on packaging tape is that they prefer customers to responsibly secure their own boxes but are willing to apply a little piece of tape once in a while out of generosity. They do not want to see their customer's boxes compromised and believe that proper communication can help customers better secure their packages. However, they no longer offer unlimited tape usage due to some customers taking advantage of the service. Each store may have a unique tape policy, as franchise-owned stores can differ in their approaches.


if __name__ == "__main__":
    main()


# Casual, Flirty, Fun, Cheerful Style:
# Hey there! We're so sorry to hear you had a less-than-stellar experience with us. While we used to provide taping services as a sweet little extra, unfortunately, some sneaky folks started taking advantage of it! We had to make a change to ensure all our packages arrive safely and securely. We totally understand the frustration of having to tape your own boxes, but we promise it's all in the name of keeping your shipments safe and sound. We appreciate your feedback and hope to make your next visit a tape-tastic one!

# Humorous, Relatable, Adventurous Style:
# Oh no, sounds like you stumbled into the wild world of taping dilemmas! We get it, nobody wants to play the role of DIY package engineer. But hey, we had to switch things up after some crafty customers started expecting us to perform box miracles. It's all about keeping your shipments snug and secure! We'll keep working on our communication to make sure you're not left feeling stuck in tape limbo. Thanks for sharing your experience with us—may your future packaging adventures be smooth sailing!

# Jovial, Practical, Trusty Neighbor Style:
# Howdy, neighbor! We're sorry to hear you weren't able to get your box taped up during your visit. We used to offer that service, but things got a bit sticky when some boxes arrived looking like they'd been through a wrestling match! We had to make a change to ensure everything stays ship-shape and secure. We understand it can be a hassle to tape up your own packages, but we promise it's all in the name of keeping your items safe and sound. Your feedback is truly appreciated, and we're here to help with any future packaging needs. Thanks for being understanding!
