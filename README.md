# Career Companion

CareerMap Chat is a lightweight conversational tool designed to extract a user’s interests through natural dialogue and map them to predefined career domains like STEM, Arts, Sports, etc. It aims to feel intuitive, friendly, and adaptive—offering meaningful career suggestions based on how someone talks about what they like, rather than relying on static questionnaires.

                  User Input > Prompt Templates > Extracted Interests > Mapped Career Path(s) > Generated Explanation(s)



# Project Flow: CareerMap Chat
Define High-Level Career Paths
A predefined set of broad career categories is established (e.g., STEM, Arts, Sports) to serve as target mappings for user interest identification.

1.Design Prompt Templates
Custom prompt templates are developed to extract user preferences from natural language inputs. These prompts encourage freeform, conversational responses and guide the language model to associate expressed interests with relevant career paths.

2.Map Interests to Career Paths
The SentenceTransformer model (all-MiniLM-L6-v2) is used to generate embeddings for both user responses and career category descriptions. Cosine similarity scoring is applied to determine the best-fit career path based on semantic alignment.

3.Psychological-Style Fallback Questions
If the user's initial responses are vague or ambiguous, a set of reflective fallback questions is triggered. These are designed to explore cognitive and emotional preferences (e.g., preferred environments, problem-solving styles, values like creativity vs. logic) to enrich the input before re-processing.

4.Generate Short Explanations
Fallback explanations for each career path are predefined. Additionally, a personalized insight is generated using the TinyLlama model (TinyLlama-1.1B-Chat-v1.0), which contextualizes the user’s input in relation to the selected career path.


# Areas I can improve

1.Deeper Data Mapping – Capture more detailed insights on values, hobbies, and strengths.

2.Percentage-Based Matching – Show confidence scores for top career paths.

3.Login & User Data – Collect basic student info for personalized tracking.
