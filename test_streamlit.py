import streamlit as st

st.title("Simple Streamlit Test App")
st.write("If you can see this, Streamlit is working correctly!")

# Display basic UI elements
st.header("Basic UI Elements")
st.subheader("Button")
if st.button("Click me"):
    st.success("Button clicked!")

st.subheader("Text Input")
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")
