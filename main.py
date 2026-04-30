from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph import add_messages
import json
import sys
from chains import generate_chain, reflect_chain

# Define the state schema for the agent's state graph using a TypedDict. 
# This schema includes the conversation history (messages), the reflection count, and the quality score from the latest reflection.
class State(TypedDict):
    messages: Annotated[list[BaseMessage] ,add_messages]
    reflection_count: Annotated[int, "The number of times the agent has reflected on the generated tweet."]
    quality_score: Annotated[int, "The quality score from the latest reflection (1-10)"]

# Read configuration from config.JSON
with open("config.JSON", "r") as f:
    config = json.load(f)
max_iterations = config.get("max_iterations", 3)
quality_threshold = config.get("quality_threshold", 8)
quality_enabled = config.get("enable_quality_check", True)

# Define the condition function to determine whether to continue reflecting or to end the process
def should_reflect(state: State) -> bool:
    '''Determine whether the agent should continue reflecting on the generated tweet or end the process. The agent should continue reflecting if the reflection count is less than the maximum allowed iterations and if the quality score is below the specified threshold (if quality check is enabled).

    Args:        state (State): The current state of the agent, including the conversation history, reflection count, and quality score.
    Returns:        bool: True if the agent should continue reflecting, False if it should end the process.
    '''
    if state["reflection_count"] >= max_iterations:
        return False
    if quality_enabled and state.get("quality_score", 0) >= quality_threshold:
        return False
    return True

def generate(state: State) -> dict:
    '''Invoke the generation chain to create a twitter post based on the user's request, reflection agent recommendations.

    Args:        state (State): The current state of the agent, including the conversation history, reflection count, and quality score.
    Returns:        dict: A dictionary containing the updated conversation history with the newly generated tweet.'''
    print("In Generator agent...")

    # invoke the generate_chain with the conversation history to generate a new tweet. The chain will use the user's request and any critiques or recommendations from the reflection agent to produce a new version of the tweet.
    generated_tweet = generate_chain.invoke(state["messages"]).content
    # print(f"==========Generated Tweet============\n {generated_tweet}")
    return {
        "messages": [AIMessage(content=generated_tweet)]
    }

def reflect(state: State) -> dict:
    '''Invoke the reflection chain to critique the generated tweet and provide a quality score. The reflection agent will analyze the generated tweet, provide detailed feedback, and assign a quality score based on the criteria defined in the reflection system prompt.

    Args:        state (State): The current state of the agent, including the conversation history, reflection count, and quality score.
    Returns:        dict: A dictionary containing the updated conversation history with the critique from the reflection agent, the updated reflection count, and the quality score from the latest reflection.'''
    print("In Reflector agent...")
    new_count = state["reflection_count"] + 1

    response = reflect_chain.invoke(state["messages"])
    import json
    result = json.loads(response.content)
    quality_score = result["quality_score"]
    critique = result["critique"]

    print(f"Reflection Count: {state['reflection_count']}, Quality Score: {quality_score}")
    return {
        "messages": [AIMessage(content=critique)],
        "reflection_count": new_count,
        "quality_score": quality_score
    }

# Initialize the state graph with the defined schema
graph_builder =StateGraph(state_schema=State)

# Add Nodes to the graph
graph_builder.add_node("generate", generate)
graph_builder.add_node("reflect", reflect)

# Add edges to the graph to define the flow of the agent
graph_builder.add_edge(START, "generate")
graph_builder.add_conditional_edges("generate", should_reflect, {
    True: "reflect",
    False: END
})
graph_builder.add_edge("reflect", "generate")

# Compile the graph to create an executable agent
graph = graph_builder.compile()

def main():
    '''Main function to run the reflection agent. This function initializes the agent's state with the user's request, invokes the graph, and prints the final output including the generated tweet, number of reflections, and final quality score.'''

    # Reconfigure stdout to ensure proper encoding for printing the output
    sys.stdout.reconfigure(encoding='utf-8')
    print("Hello from reflection-agent!")
    # Get user input
    user_request = input("Please enter your request for a twitter post: ")
    # Initialze the graph state with user input and count to 0
    initial_state: State = {
    "messages": [HumanMessage(content=user_request)],
    "reflection_count": 0,
    "quality_score": 0
    }    
    # Invoke the graph with the initial state
    final_state = graph.invoke(initial_state)
    output_tweet = final_state["messages"][-1].content  # Assuming the final output tweet is the last message in the conversation history
    print("=== Final Output ===")
    print(f"User Request: {user_request}")
    print(f"Number of Reflections: {final_state['reflection_count']}")
    print(f"Final Quality Score: {final_state.get('quality_score', 'N/A')}")
    print(f"Final Generated Tweet: {output_tweet}")

if __name__ == "__main__":
    main()
