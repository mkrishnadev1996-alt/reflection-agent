from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from llm import llm

class ReflectionResponse(BaseModel):
    quality_score: int = Field(description="Quality score of the tweet on a scale of 1 to 10")
    critique: str = Field(description="Detailed critique and recommendations for improving the tweet")

reflection_prompt = ChatPromptTemplate.from_messages(
    [
       SystemMessage(content = "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet. Always provide detailed recommendations, including requests for length, virality, style, etc. IMPORTANT: You must respond with valid JSON only, containing two fields: 'quality_score' (integer 1-10) and 'critique' (your detailed feedback)."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content ="You are a twitter techie influencer assistant tasked with writing excellent twitter posts. Generate the best twitter post possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

if __name__ == "__main__":
    # Test the generation chain
    test_messages = [SystemMessage(content="User request: Write a tweet about AI.")]
    generation_response = generate_chain.invoke(test_messages)
    print("=== Generation Chain Response ===")
    # print(generation_response)
    print(generation_response.content)  # Assuming the generated tweet is the last message in the response

    # Test the reflection chain
    reflection_response = reflect_chain.invoke(
        {
            "messages": test_messages + [HumanMessage(content=generation_response.content)]
            }
            )

    print("=== Reflection Chain Response ===")
    print(reflection_response.content)  # Assuming the generated tweet is the last message in the response
