import json
import os

# File path for saved reviews
REVIEWS_FILE = "saved_reviews.json"

# customer_review = """The only place I've been to that consistently refuses to tape your shipment for you even though it costs them less than a cent, and instead force you to buy their outrageously overpriced tape. Again, either because they're too lazy to put a piece of tape for you on a box or they get a kickback for scamming customers being forced to buy their $10 tape."""
# customer_review = """I can't believe that I have to pay $5 dollars to print a single page for my return label.  These guys are such scammers. You know the public library has printing for free!"""
# customer_review = """This store is too busy! Too many people come here to drop off returns! What the heck! I just want to send a package via next day! What the heck?"""


def display_review_management(st, callback_on_select=None):
    if "customer_review" not in st.session_state:
        st.session_state.customer_review = ""

    # Load reviews
    st.session_state.saved_reviews = load_reviews()

    review_tab1, review_tab2 = st.tabs(["New Review", "Saved Reviews"])

    with review_tab1:
        # User input
        customer_review = st.text_area(
            "2. Enter the customer review:",
            value=st.session_state.customer_review,
            height=100,
            key="review_input",
        )

        st.session_state.customer_review = customer_review

        # Save button
        if st.button("Save Review", key="save_new") and customer_review:
            st.session_state.saved_reviews.append(customer_review)
            save_review(customer_review)
            st.success("Review saved!")

    with review_tab2:
        if not st.session_state.saved_reviews:
            st.info("No saved reviews yet.")
        else:
            # Create a selectbox for review selection
            selected_index = st.selectbox(
                "Select a saved review",
                range(len(st.session_state.saved_reviews)),
                format_func=lambda i: st.session_state.saved_reviews[i],
                key="review_selector",  # Important: give it a try
            )

            # Use buttons with callbacks
            if st.button("Use Selected Review", key="use_btn"):
                selected_review = st.session_state.saved_reviews[selected_index]
                st.session_state.customer_review = selected_review
                if callback_on_select:
                    callback_on_select(selected_review)

            if st.button("Delete Selected Review", key="delete_btn"):
                st.session_state.saved_reviews.pop(selected_index)
                delete_review(selected_index)
                st.success("Review deleted!")


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
