import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.globals import set_debug
from langchain_core.globals import set_llm_cache

# from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI

import few_shot
import rag
import reviews

set_llm_cache(None)
# set_llm_cache(InMemoryCache())

# Load environment variables from .env
load_dotenv()
set_debug(False)
# set_debug(True)


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

    upload_and_ingest_file(st)
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
                response, output_parser = generate_content()
                display_content(response, output_parser)

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


def upload_and_ingest_file(st):
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
                        st.session_state.retriever = rag.create_retriever(tmp_file_path)
                        st.success("Successfully processed and ingested pdf")
                        os.unlink(tmp_file_path)

                    except Exception as e:
                        # Clean up the temporary file
                        os.unlink(tmp_file_path)

                        st.error(f"Error processing PDF: {e}")
                        st.session_state.retriever = None
                        st.stop()


def generate_content():
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
    output_parser = few_shot.create_output_parser()
    few_shot_prompt = few_shot.load_few_shot_prompt(output_parser)

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
    print(f"Generate response, customer review is {st.session_state.customer_review}")

    response = qa_chain.invoke(
        {"query": f"{st.session_state.customer_review} [ts:{time.time()}]"}
    )

    return response, output_parser


def display_content(response, output_parser):
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

        tab1, tab2, tab3 = st.tabs(["Empathetic", "Light-hearted", "Neighborly"])

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


if __name__ == "__main__":
    main()
