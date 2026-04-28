from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END, START
import json
from chains import generate_chain, reflect_chain

# Define the structure of the agent's state
class State(TypedDict):
    messages: Annotated[list[BaseMessage] ,"The conversation history between the user and the agent."]
    reflection_count: Annotated[int, "The number of times the agent has reflected on the generated tweet."]


def should_reflect(state: State) -> bool:
    '''Determine whether the agent should reflect on the generated tweet based on the conversation history and the number of reflections already performed.'''
    # Simple heuristic: reflect if there are less than  max_iteration from the config.json file messages in the conversation history
    # Get max_iterations from the config file
    with open("config.JSON", "r") as f:
        config = json.load(f)
    max_iterations = config.get("max_iterations", 5)  # Default to 5

    return state["reflection_count"] < max_iterations

def generate(state: State) -> State:
    '''Invoke the generation chain to create a twitter post based on the user's request, reflection agent recommendations.'''
    print("In Generator agent...")
    response = generate_chain.invoke(state["messages"])
    generated_tweet = response.content
    print(f"Generated Tweet: {generated_tweet}")
    state["messages"].append(AIMessage(content=generated_tweet))
    return state

def reflect(state: State) -> State:
    '''Invoke the reflection chain to critique the generated tweet and provide recommendations.'''
    print("In Reflector agent...")
    state["reflection_count"] += 1
    response = reflect_chain.invoke(state["messages"])
    critique = response.content
    print(f"Critique and Recommendations: {critique}")
    state["messages"].append(AIMessage(content=critique))
    return state

# Initialize the state graph with the defined schema
graph =StateGraph(state_schema=State)

# Add Nodes to the graph
graph.add_node("generate", generate)
graph.add_node("reflect", reflect)

# Add edges to the graph to define the flow of the agent
graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", should_reflect, {
    True: "reflect",
    False: END
})
graph.add_edge("reflect", "generate")
graph_compiled = graph.compile()

def main():
    print("Hello from reflection-agent!")
    # get user input
    user_request = input("Please enter your request for a twitter post: ")
    initial_state: State = {
    "messages": [HumanMessage(content=user_request)],
    "reflection_count": 0
    }    
    # Run the graph with the initial state
    final_state = graph_compiled.invoke(initial_state)
    output_tweet = final_state["messages"][-1].content  # Assuming the final output tweet is the last message in the conversation history
    print("=== Final Output ===")
    print(f"User Request: {user_request}")
    print(f"Number of Reflections: {final_state['reflection_count']}")
    print(f"Final Generated Tweet: {output_tweet}")

if __name__ == "__main__":
    main()
