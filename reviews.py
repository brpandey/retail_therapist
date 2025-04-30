import json
import os

# File path for saved reviews
REVIEWS_FILE = "saved_reviews.json"

from langchain.globals import set_llm_cache

set_llm_cache(None)  # Disable caching


def display_review_management(st, callback_on_select=None):
    if "customer_review" not in st.session_state:
        st.session_state.customer_review = ""

    # Load reviews
    st.session_state.saved_reviews = load_reviews()

    review_tab1, review_tab2 = st.tabs(["New Review", "Saved Reviews"])

    with review_tab1:

        def on_review_entered():
            # st.session_state.customer_review = customer_review
            st.session_state.customer_review = st.session_state.review_input
            print(f"on_review_entered: {st.session_state.review_input}")

        # User input
        customer_review = st.text_area(
            "2. Enter the customer review:",
            value=st.session_state.customer_review,
            height=100,
            key="review_input",
            on_change=on_review_entered,
        )

        # Save button
        if (
            st.button("Save Review", key="save_new", type="tertiary")
            and customer_review
        ):
            st.session_state.saved_reviews.append(customer_review)
            save_review(customer_review)
            st.success("Review saved!")

    with review_tab2:
        if not st.session_state.saved_reviews:
            st.info("No saved reviews yet.")
        else:
            refresh = False

            # Define callback function for when selection changes
            def on_review_selected():
                # Get the selected review
                selected_review = st.session_state.saved_reviews[
                    st.session_state.review_selector
                ]
                # Update the customer review in session state
                st.session_state.customer_review = selected_review

            # Create a selectbox for review selection
            selected_index = st.selectbox(
                "Select a saved review",
                range(len(st.session_state.saved_reviews)),
                format_func=lambda i: st.session_state.saved_reviews[i],
                key="review_selector",
                on_change=on_review_selected,
            )

            if st.button("Delete Selected Review", key="delete_btn", type="tertiary"):
                st.session_state.saved_reviews.pop(selected_index)
                delete_review(selected_index)
                st.success("Review deleted!")
                refresh = True

            if refresh:
                st.rerun()


# Load saved reviews
def load_reviews():
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Handle case of corrupt file
                return []
    return []


# Save a review
def save_review(review):
    reviews = load_reviews()

    # Check if review already exists to avoid duplicates
    if not any(r == review for r in reviews):
        reviews.append(review)
        with open(REVIEWS_FILE, "w") as f:
            json.dump(reviews, f)
        return True
    return False


# Delete a review by ID
def delete_review(review_id):
    reviews = load_reviews()
    reviews.pop(review_id)

    with open(REVIEWS_FILE, "w") as f:
        json.dump(reviews, f)
