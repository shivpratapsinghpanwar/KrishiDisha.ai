#!/usr/bin/env python3
"""
A simple rule-based chatbot
"""

import random

class SimpleChatbot:
    def __init__(self):
        self.name = "SimpleBot"
        self.responses = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! Nice to meet you!",
            ],
            "how_are_you": [
                "I'm doing great, thanks for asking! How about you?",
                "I'm good! How are you doing?",
                "Doing well, thank you!",
            ],
            "name": [
                f"My name is {self.name}.",
                f"I'm {self.name}, your friendly chatbot!",
            ],
            "bye": [
                "Goodbye! Have a great day!",
                "See you later!",
                "Bye! Take care!",
            ],
            "thanks": [
                "You're welcome!",
                "Happy to help!",
                "No problem at all!",
            ],
            "default": [
                "I'm not sure I understand. Can you rephrase that?",
                "Interesting! Tell me more.",
                "I'm still learning. Could you say that differently?",
            ],
        }
        
        self.keywords = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "how_are_you": ["how are you", "how's it going", "how do you do"],
            "name": ["what's your name", "who are you", "your name"],
            "bye": ["bye", "goodbye", "see you", "exit", "quit"],
            "thanks": ["thank", "thanks", "appreciate"],
        }
    
    def get_response(self, user_input):
        """Process user input and return an appropriate response."""
        user_input = user_input.lower().strip()
        
        # Check for keywords in the input
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in user_input:
                    return random.choice(self.responses[category])
        
        # Default response if no keywords match
        return random.choice(self.responses["default"])
    
    def chat(self):
        """Start a conversation with the user."""
        print(f"{self.name}: Hello! I'm {self.name}. Type 'bye' to exit.")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                response = self.get_response(user_input)
                print(f"{self.name}: {response}")
                
                # Check if user wants to exit
                if any(word in user_input.lower() for word in ["bye", "goodbye", "exit", "quit"]):
                    break
                    
            except KeyboardInterrupt:
                print(f"\n{self.name}: Goodbye!")
                break
            except EOFError:
                break


if __name__ == "__main__":
    bot = SimpleChatbot()
    bot.chat()
