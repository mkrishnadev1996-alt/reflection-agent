from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage 
from langgraph.graph import StateGraph, END, START
from langgraph.graph import add_messages
import json
from chains import generate_chain, reflect_chain

# Define the structure of the agent's state
class State(TypedDict):
    # The conversation history between the user and the agent, including generated tweets and critiques.   
    # add_messages is a helper function that ensures the list of messages is properly formatted for the graph's state management. It allows us to easily append new messages to the conversation history while maintaining the correct structure.
    messages: Annotated[list[BaseMessage] ,add_messages] 
    reflection_count: Annotated[int, "The number of times the agent has reflected on the generated tweet."]

# Get max_iterations from the config file
with open("config.JSON", "r") as f:
    config = json.load(f)
max_iterations = config.get("max_iterations", 3)  # Default to 3 if not specified
    
def should_reflect(state: State) -> bool:
    '''Determine whether the agent should reflect on the generated tweet based on the conversation history and the number of reflections already performed.'''
    # Simple heuristic: reflect if there are less than  max_iteration from the config.json file messages in the conversation history  
    # print(state["reflection_count"] < max_iterations)

    return state["reflection_count"] < max_iterations

def generate(state: State) -> dict:
    '''Invoke the generation chain to create a twitter post based on the user's request, reflection agent recommendations.'''
    print("In Generator agent...")

    # invoke the generate_chain with the conversation history to generate a new tweet. The chain will use the user's request and any critiques or recommendations from the reflection agent to produce a new version of the tweet.
    generated_tweet = generate_chain.invoke(state["messages"]).content
    # print(f"==========Generated Tweet============\n {generated_tweet}")
    return {
        "messages": [AIMessage(content=generated_tweet)]
    }

def reflect(state: State) -> State:
    '''Invoke the reflection chain to critique the generated tweet and provide recommendations.'''
    print("In Reflector agent...")
    new_count = state["reflection_count"] + 1

    # Invoke the reflect_chain with the conversation history to critique the generated tweet and provide recommendations for improvement. The chain will analyze the generated tweet in the context of the user's original request and any previous critiques to offer constructive feedback.
    critique = reflect_chain.invoke(state["messages"]).content
    # print(f"===============Critique and Recommendations=========== \n{critique}")
    print(f"Reflection Count: {state['reflection_count']}")
    return {
        "messages": [AIMessage(content=critique)],
        "reflection_count": new_count
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
    print("Hello from reflection-agent!")
    # Get user input
    user_request = input("Please enter your request for a twitter post: ")
    # Initialze the graph state with user input and count to 0
    initial_state: State = {
    "messages": [HumanMessage(content=user_request)],
    "reflection_count": 0
    }    
    # Invoke the graph with the initial state
    final_state = graph.invoke(initial_state)
    output_tweet = final_state["messages"][-1].content  # Assuming the final output tweet is the last message in the conversation history
    print("=== Final Output ===")
    print(f"User Request: {user_request}")
    print(f"Number of Reflections: {final_state['reflection_count']}")
    print(f"Final Generated Tweet: {output_tweet}")

if __name__ == "__main__":
    main()
