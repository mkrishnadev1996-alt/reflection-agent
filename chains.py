from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from llm import llm
from prompts import GENERATION_PROMPT, REFLECTION_PROMPT

# Pydantic model for parsing structured JSON responses from the reflection chain
# The reflection chain returns JSON with quality_score (1-10) and critique (text feedback)
class ReflectionResponse(BaseModel):
    quality_score: int = Field(description="Quality score of the tweet on a scale of 1 to 10")
    critique: str = Field(description="Detailed critique and recommendations for improving the tweet")

# Chains combine prompt templates with the LLM to create reusable, invokeable units
# Generation chain: takes user request + conversation history -> generates tweet
generate_chain = GENERATION_PROMPT | llm

# Reflection chain: takes conversation history -> critiques tweet and provides quality score
# Uses PydanticOutputParser, bind_tools to ensure the LLM's response is valid Pydantic object
# that can be parsed into the ReflectionResponse model
reflect_chain = REFLECTION_PROMPT | llm.bind_tools(tools=[ReflectionResponse]) | PydanticOutputParser(pydantic_object=ReflectionResponse)

if __name__ == "__main__":
    # Test script to verify chains are working correctly
    from langchain_core.messages import SystemMessage, HumanMessage

    test_messages = [SystemMessage(content="User request: Write a tweet about AI.")]
    generation_response = generate_chain.invoke(test_messages)
    print("=== Generation Chain Response ===")
    print(generation_response.content)

    reflection_response = reflect_chain.invoke(
        {
            "messages": test_messages + [HumanMessage(content=generation_response.content)]
        }
    )

    print("=== Reflection Chain Response ===")
    print(reflection_response.content)