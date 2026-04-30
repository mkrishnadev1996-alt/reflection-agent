from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# System prompt for the reflection chain - instructs the LLM to act as a viral twitter influencer
# grading tweets and providing JSON-formatted critique with quality_score and critique fields
REFLECTION_SYSTEM_PROMPT = """You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet. Always provide detailed recommendations, including requests for length, virality, style, etc. IMPORTANT: You must respond with valid JSON only, containing two fields: 'quality_score' (integer 1-10) and 'critique' (your detailed feedback)."""

# ChatPromptTemplate for the reflection chain - combines system prompt with conversation history via MessagesPlaceholder
REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

# System prompt for the generation chain - instructs the LLM to act as a twitter techie influencer
# tasked with writing excellent twitter posts, revising based on any critique provided
GENERATION_SYSTEM_PROMPT = """You are a twitter techie influencer assistant tasked with writing excellent twitter posts. Generate the best twitter post possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts."""

# ChatPromptTemplate for the generation chain - combines system prompt with conversation history via MessagesPlaceholder
GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=GENERATION_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])