import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- ChromaDB Initialization (Keep as is) ---
# Initialize embedding function and chroma client
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client(Settings(chroma_db_impl="duckdb"))



# Delete existing collection if exists
try:
    # Ensure a fresh collection for each run to avoid stale data from previous uploads
    chroma_client.delete_collection(name="data_cleaning")
    st.info("Previous 'data_cleaning' collection deleted.")
except Exception as e:
    st.info(f"Info: Could not delete existing collection (might not exist yet or an error occurred). Reason: {e}")

# Create new collection
collection = chroma_client.create_collection(name="data_cleaning", embedding_function=embedding_fn)

st.set_page_config(page_title="Data Cleaning Assistant with RAG Bot")
st.title(" Data Cleaning Assistant with RAG Bot")
st.markdown("Upload your CSV file to get insights, cleaning suggestions, and ask questions about your data!")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Define metadata generation function outside of the if uploaded_file block
# so it's accessible for RAG later even if not directly displayed.
def generate_meta_data(df):
    return {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "num_rows": len(df),
        "unique_values": df.nunique().to_dict(),
    }

# Convert metadata to text for RAG
def metadata_to_query(df):
    query_parts = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isnull().sum()
        
        mean_val, median_val, std_val, skew_val, kurt_val = "N/A", "N/A", "N/A", "N/A", "N/A"

        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                mean_val = round(df[col].mean(), 2)
                median_val = round(df[col].median(), 2) # Added median
                std_val = round(df[col].std(), 2)
                skew_val = round(df[col].skew(), 2)
                kurt_val = round(df[col].kurtosis(), 2)
            except Exception:
                pass # Still catch errors if calculation fails for some reason

        query_parts.append(
            f"{col}: type={dtype}, missing={missing}, mean={mean_val}, median={median_val}, std={std_val}, skew={skew_val}, kurtosis={kurt_val}"
        )
    return "\n".join(query_parts)


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df # Store DataFrame in session state for the bot

    st.subheader("Dataset Preview")
    st.write(df.head())

    meta_data = generate_meta_data(df)

    # Create a metadata table
    meta_table = pd.DataFrame({
        "Column": meta_data["columns"],
        "Data Type": list(meta_data["dtypes"].values()),
        "Missing Values": list(meta_data["missing_values"].values()),
        "Unique Values": list(meta_data["unique_values"].values())
    })

    st.subheader("Dataset Metadata")
    st.markdown(f"**Total Rows in Dataset:** `{meta_data['num_rows']}`")
    st.dataframe(meta_table)

    # Numerical analysis
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    stats_df = pd.DataFrame() # Initialize an empty DataFrame
    if numeric_cols:
        st.subheader("Statistical Summary & Outlier Detection")

        stats_df = df[numeric_cols].agg(
            ['mean', 'median', 'std', 'skew', 'kurtosis']
        ).transpose().reset_index().rename(columns={'index': 'Column'})
        stats_df = stats_df.round(3)

        def detect_outliers_iqr(col):
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return ((col < lower) | (col > upper)).sum()

        def detect_outliers_z(col):
            # Using dropna() to avoid issues with NaNs in zscore calculation
            return (np.abs(zscore(col.dropna())) > 3).sum()

        stats_df["IQR Outliers"] = [detect_outliers_iqr(df[col]) for col in stats_df["Column"]]
        stats_df["Z-Score Outliers"] = [detect_outliers_z(df[col]) for col in stats_df["Column"]]

        st.dataframe(stats_df)

        st.subheader("Distribution & Boxplot (per Numeric Column)")
        for col in numeric_cols:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            sns.histplot(df[col], kde=True, ax=axs[0], color="skyblue")
            axs[0].set_title(f'Distribution of {col}')

            sns.boxplot(x=df[col], ax=axs[1], color="salmon")
            axs[1].set_title(f'Boxplot of {col}')

            st.pyplot(fig)
            plt.close(fig) # Close the figure to free up memory

    # --- Data Cleaning Suggestions Section ---
    st.subheader("🧹 Data Cleaning Suggestions")
    st.markdown("Based on the analysis, here are some recommended cleaning actions:")

    # 1. Missing Values Suggestions
    missing_cols = [col for col, count in meta_data["missing_values"].items() if count > 0]
    if missing_cols:
        st.markdown("##### 1. Handle Missing Values")
        for col in missing_cols:
            missing_count = meta_data["missing_values"][col]
            missing_percentage = (missing_count / meta_data["num_rows"]) * 100
            st.write(f"- Column `{col}` has `{missing_count}` missing values ({missing_percentage:.2f}%).")
            if missing_percentage < 5:
                st.info(f"  *Suggestion:* Consider `imputing` missing values (e.g., with mean/median for numerical, mode for categorical) or `dropping` rows for `{col}`.")
            elif missing_percentage < 30:
                st.warning(f"  *Suggestion:* For `{col}`, `imputation` (e.g., mean, median, or more advanced methods like regression imputation) is generally preferred. `Dropping rows` might lead to significant data loss.")
            else:
                st.error(f"  *Suggestion:* With `{missing_percentage:.2f}%` missing values in `{col}`, consider if the column is useful. You might need to `drop the column` or use `advanced imputation techniques` with caution.")
    else:
        st.info("No columns with missing values found. Good job! 🎉")

    # 2. Outlier Suggestions
    if not stats_df.empty:
        outlier_cols_iqr = stats_df[stats_df["IQR Outliers"] > 0]["Column"].tolist()
        outlier_cols_z = stats_df[stats_df["Z-Score Outliers"] > 0]["Column"].tolist()

        if outlier_cols_iqr or outlier_cols_z:
            st.markdown("##### 2. Address Outliers")
            for col in set(outlier_cols_iqr + outlier_cols_z): # Use set to avoid duplicates
                iqr_outliers = stats_df[stats_df["Column"] == col]["IQR Outliers"].iloc[0]
                z_outliers = stats_df[stats_df["Column"] == col]["Z-Score Outliers"].iloc[0]
                st.write(f"- Column `{col}` shows `{int(iqr_outliers)}` IQR outliers and `{int(z_outliers)}` Z-score outliers.")
                st.info(f"  *Suggestion:* Investigate these outliers. Options include: `removing` them (if data entry errors), `transforming` the data (e.g., log transform for skewed data), `capping/winsorizing` them, or using `robust models` that are less sensitive to outliers.")
        else:
            st.info("No significant outliers detected in numerical columns based on IQR and Z-score (threshold > 3).")
    else:
        st.info("No numerical columns to check for outliers.")

    # 3. Data Type Suggestions (Common issues)
    st.markdown("##### 3. Review Data Types")
    type_issues_found = False
    for col, dtype in meta_data["dtypes"].items():
        # Check for numeric columns that might be objects (e.g., '123,45', '$100')
        if dtype == 'object' and any(char.isdigit() for char in df[col].dropna().astype(str).unique()):
            try:
                # Try converting to numeric, if it fails, it's not straightforward
                # Using regex to clean common non-numeric chars like '$' and ','
                pd.to_numeric(df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='raise')
                st.warning(f"- Column `{col}` is `object` type but appears to contain numeric values (e.g., '1,234' or '$50').")
                st.info(f"  *Suggestion:* Consider `converting to numeric` after removing non-numeric characters (e.g., `df['{col}'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)`).")
                type_issues_found = True
            except ValueError:
                pass # Not a simple numeric conversion issue

        # Check for date-like strings that are objects
        if dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().any():
            # Exclude purely numeric columns that might parse as dates if they are truly numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(f"- Column `{col}` is `object` type but contains values that look like dates.")
                st.info(f"  *Suggestion:* Convert to `datetime` objects using `pd.to_datetime(df['{col}'])` for proper time-series analysis.")
                type_issues_found = True

    if not type_issues_found:
        st.info("Data types appear generally appropriate or require specific domain knowledge for further review.")

    # 4. High Cardinality Categorical Columns
    st.markdown("##### 4. High Cardinality Categorical Columns")
    high_cardinality_found = False
    # Only consider object or category dtypes for cardinality
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = meta_data["unique_values"][col]
        total_rows = meta_data["num_rows"]
        # Heuristic: more than 50% unique values and at least 20 unique values
        if unique_count / total_rows > 0.5 and unique_count > 20:
            st.warning(f"- Column `{col}` is a `categorical` type with `{unique_count}` unique values (high cardinality).")
            st.info(f"  *Suggestion:* High cardinality can be problematic for some models. Consider `feature engineering` (e.g., grouping rare categories, target encoding, or applying embedding techniques) or `dropping the column` if not highly relevant.")
            high_cardinality_found = True
    if not high_cardinality_found:
        st.info("No categorical columns identified with unusually high cardinality that typically pose issues for modeling.")


    # Store metadata in ChromaDB for RAG-style QA
    st.subheader("🤖 Ask Your Data Bot")
    query_text = metadata_to_query(df)

    # Always delete and re-add to ensure the most current metadata is in ChromaDB
    # This prevents old data's metadata from interfering if a new file is uploaded
    try:
        collection.delete(ids=["summary_001"])
    except Exception as e:
        # st.warning(f"Could not delete old summary_001: {e}") # This might be normal if it's the first upload
        pass # Silently pass if ID doesn't exist

    collection.add(
        documents=[query_text],
        metadatas=[{"source": "data_summary"}],
        ids=["summary_001"]
    )
    st.success("Dataset metadata has been processed and stored for bot queries.")


    # --- RAG User Query Bot ---
    user_question = st.text_input("Ask a question about your dataset (e.g., 'What columns have missing values?', 'Tell me about outliers in numeric columns?', 'What is the data type of the Age column?', 'What is the mean and median of the Salary column?', 'Explain outcome and explanatory variables in this dataset.')")

    if user_question:
        # 1. Retrieval
        st.info("Searching for relevant information...")
        results = collection.query(
            query_texts=[user_question],
            n_results=1 # We expect the single summary document to be most relevant
        )

        retrieved_document = results['documents'][0][0] if results['documents'] else "No relevant information found."
        # st.write(f"**Retrieved Context:**\n```\n{retrieved_document}\n```") # For debugging, uncomment this line


        # 2. Augmentation & Generation (Simulated LLM)
        prompt = f"""
        You are a helpful data analysis assistant. Based on the provided dataset metadata and general data science knowledge, answer the user's question.
        If the answer is not directly available in the metadata, state that you cannot provide a precise answer based on the given context.

        Definitions:
        - An **Outcome Variable** (or dependent/response variable) is the variable whose variation you are trying to understand or predict. It is the "effect."
        - An **Explanatory Variable** (or independent/predictor variable) is a variable that is thought to influence or explain changes in the outcome variable. It is the "cause" or predictor.
        Identifying these often depends on the specific research question.

        Dataset Metadata:
        ```
        {retrieved_document}
        ```

        User Question: {user_question}

        Answer:
        """

        llm_response = "I couldn't find a specific answer in the provided metadata." # Default response
        user_q_lower = user_question.lower()

        # Handle "overall summary"
        if "overall summary" in user_q_lower or "summarize" in user_q_lower or "tell me about the dataset" in user_q_lower:
            llm_response = f"Here is an overall summary of your dataset's metadata:\n\n{retrieved_document}"

        # Handle "mean", "median", "skewness", "kurtosis" for a specific column
        elif any(stat in user_q_lower for stat in ["mean", "median", "skew", "kurtosis", "std"]):
            col_name_match = None
            for col in numeric_cols: # Only check numeric columns for these stats
                if col.lower() in user_q_lower:
                    col_name_match = col
                    break
            
            if col_name_match and not stats_df.empty and col_name_match in stats_df["Column"].values:
                col_stats = stats_df[stats_df["Column"] == col_name_match].iloc[0]
                response_parts = []
                if "mean" in user_q_lower and 'mean' in col_stats:
                    response_parts.append(f"Mean: {col_stats['mean']}")
                if "median" in user_q_lower and 'median' in col_stats:
                    response_parts.append(f"Median: {col_stats['median']}")
                if "skew" in user_q_lower and 'skew' in col_stats:
                    response_parts.append(f"Skewness: {col_stats['skew']}")
                if "kurtosis" in user_q_lower and 'kurtosis' in col_stats:
                    response_parts.append(f"Kurtosis: {col_stats['kurtosis']}")
                if "std" in user_q_lower and 'std' in col_stats:
                    response_parts.append(f"Standard Deviation: {col_stats['std']}")
                
                if response_parts:
                    llm_response = f"For the `{col_name_match}` column: " + ", ".join(response_parts) + "."
                else:
                    llm_response = f"I found the `{col_name_match}` column but couldn't retrieve the specific statistical measure you asked for."
            elif col_name_match:
                llm_response = f"The column `{col_name_match}` is not numeric, so these statistics are not applicable."
            else:
                llm_response = "Please specify a numeric column for which you want to know the statistics (e.g., 'mean of Salary')."

        # Handle "missing values"
        elif "missing values" in user_q_lower:
            missing_cols_info = []
            for col, count in meta_data["missing_values"].items():
                if count > 0:
                    missing_cols_info.append(f"{col}: {count} missing values")
            if missing_cols_info:
                llm_response = "Columns with missing values are: " + ", ".join(missing_cols_info) + "."
            else:
                llm_response = "There are no columns with missing values in the dataset."
        
        # Handle "outliers"
        elif "outliers" in user_q_lower and numeric_cols and not stats_df.empty:
            outlier_details = []
            for col in set(outlier_cols_iqr + outlier_cols_z):
                iqr = stats_df[stats_df["Column"] == col]["IQR Outliers"].iloc[0]
                z = stats_df[stats_df["Column"] == col]["Z-Score Outliers"].iloc[0]
                if iqr > 0 or z > 0:
                    outlier_details.append(f"{col} (IQR: {int(iqr)}, Z-score: {int(z)})")
            if outlier_details:
                llm_response = "Columns with detected outliers are: " + ", ".join(outlier_details) + ". These are based on IQR and Z-score methods."
            else:
                llm_response = "No significant outliers were detected in numerical columns."
        
        # Handle "data type"
        elif "data type" in user_q_lower or "dtype" in user_q_lower:
            col_name_match = None
            for col in df.columns:
                if col.lower() in user_q_lower:
                    col_name_match = col
                    break
            if col_name_match:
                llm_response = f"The data type of the `{col_name_match}` column is `{meta_data['dtypes'][col_name_match]}`."
            else:
                all_dtypes = [f"{col}: {dtype}" for col, dtype in meta_data["dtypes"].items()]
                llm_response = "Here are the data types for all columns: " + ", ".join(all_dtypes) + "."
        
        # Handle "number of rows"
        elif "number of rows" in user_q_lower or "rows in dataset" in user_q_lower:
            llm_response = f"The dataset has `{meta_data['num_rows']}` rows."
        
        # Handle "columns"
        elif "columns" in user_q_lower and "number" not in user_q_lower:
            llm_response = "The columns in the dataset are: " + ", ".join(meta_data["columns"]) + "."
        
        # Handle "outcome variables" and "explanatory variables"
        elif "outcome variables" in user_q_lower or "explanatory variables" in user_q_lower or "dependent independent" in user_q_lower or "response predictor" in user_q_lower:
            numerical_cols_str = ", ".join([f"`{c}`" for c in numeric_cols]) if numeric_cols else "None"
            categorical_cols_str = ", ".join([f"`{c}`" for c in df.select_dtypes(include=['object', 'category']).columns]) if df.select_dtypes(include=['object', 'category']).columns.tolist() else "None"

            llm_response = (
                "Identifying outcome and explanatory variables requires domain knowledge and a specific research question. "
                "The bot cannot automatically determine these for your dataset.\n\n"
                "**Definition:**\n"
                "- An **Outcome Variable** (dependent/response) is what you are trying to predict or explain.\n"
                "- An **Explanatory Variable** (independent/predictor) is what you use to influence or predict the outcome.\n\n"
                "**Considerations for your dataset:**\n"
                f"- **Numerical Columns** like {numerical_cols_str} could potentially be outcome or explanatory variables.\n"
                f"- **Categorical Columns** like {categorical_cols_str} are often explanatory variables.\n\n"
                "Please formulate your research question and identify them yourself based on your understanding of the data."
            )
        else:
             # Fallback to a very basic LLM simulation if no specific keyword matched
             # This part now relies more on the direct retrieval for generic questions
             if retrieved_document != "No relevant information found." and len(retrieved_document) > 50: # Ensure it's not empty/default message
                 # Try to give a slightly more helpful general response by referencing the context
                 llm_response = "I can provide details from the dataset metadata. Please ask about specific columns, missing values, data types, or statistical summaries (like mean, median, skewness for numeric columns). "
                 if "about" in user_q_lower or "tell me" in user_q_lower:
                     llm_response += "You can also ask for an 'overall summary'."
             else:
                 llm_response = "I couldn't find a specific answer in the provided metadata. Please try rephrasing your question or ask about known aspects like 'missing values', 'data types', or 'statistics for a column'."

        st.markdown(f"**Bot's Answer:** {llm_response}")

else:
    st.info("Please upload a CSV file to begin the analysis and enable the RAG bot.")

st.sidebar.header("About")
st.sidebar.info(
    "This application helps you get a quick overview of your dataset's quality "
    "and provides actionable suggestions for data cleaning. "
    "It also includes a RAG bot that can answer questions about your data's metadata, "
    "powered by ChromaDB and a simulated LLM."
)
