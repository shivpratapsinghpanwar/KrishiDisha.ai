#!/usr/bin/env python3
"""
KrishiDisha Agriculture LLM Chatbot
A specialized chatbot for agricultural queries and farmer assistance
"""

import random
import re
from typing import Dict, List, Optional

class KrishiDishaBot:
    """Agriculture-focused chatbot for KrishiDisha project"""
    
    def __init__(self):
        self.name = "KrishiDisha"
        self.version = "1.0"
        
        # Agricultural knowledge base
        self.knowledge_base = {
            "greeting": {
                "keywords": ["hello", "hi", "hey", "namaste", "greetings", "good morning", "good evening"],
                "responses": [
                    "Namaste! Welcome to KrishiDisha. How can I help you with your farming needs today?",
                    "Hello farmer! I'm here to assist you with crop recommendations, disease identification, and more.",
                    "Greetings! Ask me anything about crops, fertilizers, or farming practices.",
                    "Welcome to KrishiDisha! Your digital agriculture assistant. What would you like to know?",
                ]
            },
            "crop_recommendation": {
                "keywords": ["crop", "plant", "grow", "cultivate", "sowing", "which crop", "what to grow"],
                "responses": [
                    "For crop recommendations, I consider factors like soil type, climate, season, and water availability. Could you tell me about your soil type and location?",
                    "I can help recommend suitable crops! Please share details about your farm's soil (N, P, K levels), temperature, and rainfall patterns.",
                    "Choosing the right crop depends on multiple factors. What's your soil type and which season are you planning to cultivate?",
                    "Based on our ML model, I can suggest the best crops for your conditions. Tell me about your soil nutrients and climate.",
                ],
                "info": {
                    "rice": "Rice grows well in clayey soil with good water retention. Requires temperatures between 20-35°C and high humidity.",
                    "wheat": "Wheat thrives in loamy soil with moderate water. Best grown in temperatures between 15-25°C during growing season.",
                    "cotton": "Cotton requires well-drained black or alluvial soil. Needs warm climate (21-30°C) and moderate rainfall.",
                    "maize": "Maize grows in various soils but prefers well-drained fertile soil. Temperature range: 18-27°C.",
                    "pulses": "Pulses like gram and lentil grow well in less fertile soils. Require moderate temperatures and less water.",
                    "vegetables": "Vegetables generally need fertile, well-drained soil with regular irrigation and moderate temperatures.",
                }
            },
            "fertilizer": {
                "keywords": ["fertilizer", "nutrient", "npk", "manure", "urea", "dap", "potash"],
                "responses": [
                    "Fertilizer selection depends on soil nutrient levels and crop requirements. Common NPK ratios vary by crop.",
                    "For balanced fertilization, test your soil first. Nitrogen (N) promotes leaf growth, Phosphorus (P) for roots, Potassium (K) for overall health.",
                    "Organic manure improves soil structure while chemical fertilizers provide quick nutrients. A combination works best.",
                    "Urea provides nitrogen, DAP gives phosphorus, and Muriate of Potash supplies potassium. Use based on soil test results.",
                ],
                "recommendations": {
                    "rice": "Apply 80-100 kg N, 40-50 kg P2O5, and 40-50 kg K2O per hectare in split doses.",
                    "wheat": "Use 120-150 kg N, 60-75 kg P2O5, and 40-60 kg K2O per hectare.",
                    "cotton": "Apply 100-120 kg N, 60-80 kg P2O5, and 60-80 kg K2O per hectare.",
                    "vegetables": "Generally require 100-150 kg N, 60-100 kg P2O5, and 80-120 kg K2O per hectare.",
                }
            },
            "disease": {
                "keywords": ["disease", "pest", "infection", "leaf spot", "wilting", "yellowing", "blight", "rust"],
                "responses": [
                    "Plant diseases can be identified by symptoms on leaves, stems, or fruits. Can you describe what you're seeing?",
                    "Common diseases include fungal infections (blight, rust), bacterial wilt, and viral mosaics. Early detection is key.",
                    "For disease management, remove affected parts, ensure proper spacing, and use appropriate fungicides or biocontrol agents.",
                    "Prevention is better than cure! Practice crop rotation, use resistant varieties, and maintain field hygiene.",
                ],
                "common_diseases": {
                    "rice_blast": "Caused by fungus Magnaporthe oryzae. Symptoms: spindle-shaped lesions on leaves. Control: Use resistant varieties, apply tricyclazole.",
                    "wheat_rust": "Fungal disease causing orange/brown pustules. Control: Spray propiconazole, use resistant varieties.",
                    "cotton_bollworm": "Pest causing damage to bolls. Control: Use Bt cotton, spray neem-based pesticides.",
                    "tomato_blight": "Late blight causes dark lesions. Control: Apply copper-based fungicides, improve air circulation.",
                }
            },
            "irrigation": {
                "keywords": ["water", "irrigation", "watering", "drought", "moisture", "rain"],
                "responses": [
                    "Proper irrigation is crucial. Different crops have different water requirements at various growth stages.",
                    "Drip irrigation saves 30-50% water compared to flood irrigation and improves yield.",
                    "Water requirements vary: Rice needs standing water, wheat needs 4-6 irrigations, millets are drought-resistant.",
                    "Monitor soil moisture before irrigating. Over-watering can be as harmful as under-watering.",
                ]
            },
            "season": {
                "keywords": ["season", "rabi", "kharif", "zaid", "monsoon", "winter", "summer"],
                "responses": [
                    "India has three main cropping seasons: Kharif (June-Oct), Rabi (Oct-March), and Zaid (March-June).",
                    "Kharif crops include rice, maize, cotton, soybean. Sown with monsoon onset.",
                    "Rabi crops include wheat, gram, mustard, peas. Sown in winter, harvested in spring.",
                    "Zaid season crops are watermelon, cucumber, vegetables. Grown between Rabi and Kharif.",
                ]
            },
            "soil": {
                "keywords": ["soil", "loam", "clay", "sandy", "black soil", "alluvial", "red soil"],
                "responses": [
                    "Soil testing is essential before cultivation. It reveals pH, NPK levels, and organic carbon content.",
                    "Clay soil retains water well but drains poorly. Sandy soil drains quickly but doesn't retain nutrients.",
                    "Loamy soil is ideal for most crops - good drainage, nutrient retention, and aeration.",
                    "Black soil (regur) is rich in calcium and magnesium, excellent for cotton and oilseeds.",
                ]
            },
            "organic": {
                "keywords": ["organic", "natural", "bio", "compost", "vermicompost", "pesticide-free"],
                "responses": [
                    "Organic farming avoids synthetic chemicals, using natural inputs like compost, green manure, and biopesticides.",
                    "Vermicompost enriches soil with beneficial microbes and improves soil structure naturally.",
                    "Neem oil, garlic spray, and cow urine are effective organic pest control methods.",
                    "Organic certification can increase market value of your produce by 20-30%.",
                ]
            },
            "market": {
                "keywords": ["market", "price", "sell", "mandi", "msp", "procurement", "apmc"],
                "responses": [
                    "Check current MSP (Minimum Support Price) announced by the government for various crops.",
                    "Register on e-NAM (National Agriculture Market) for better price discovery and wider market access.",
                    "Consider forming FPOs (Farmer Producer Organizations) for better bargaining power.",
                    "Store produce properly to avoid distress sales. Use warehousing facilities when prices are low.",
                ]
            },
            "government_schemes": {
                "keywords": ["scheme", "subsidy", "loan", "pmkisan", "insurance", "support"],
                "responses": [
                    "PM-KISAN provides ₹6,000 per year in three installments to eligible farmer families.",
                    "Pradhan Mantri Fasal Bima Yojana (PMFBY) offers crop insurance at subsidized premiums.",
                    "Kisan Credit Card (KCC) provides easy credit for agricultural activities at concessional rates.",
                    "Subsidies available on seeds, fertilizers, farm machinery, and irrigation equipment.",
                ]
            },
            "goodbye": {
                "keywords": ["bye", "goodbye", "thank", "thanks", "exit", "quit", "see you"],
                "responses": [
                    "Thank you for using KrishiDisha! Wishing you a bountiful harvest. Jai Kisan!",
                    "Goodbye! Feel free to return anytime for farming advice. Happy farming!",
                    "You're welcome! Remember, sustainable farming leads to prosperous farming. Take care!",
                    "Namaste! May your fields flourish and bring prosperity. Visit us again!",
                ]
            }
        }
        
        # Conversation context
        self.context = {
            "user_name": None,
            "location": None,
            "farm_size": None,
            "current_crop": None,
            "issue_type": None
        }
        
        self.last_topic = None
    
    def extract_info(self, user_input: str) -> Dict[str, Optional[str]]:
        """Extract relevant information from user input"""
        info = {}
        user_input_lower = user_input.lower()
        
        # Extract location mentions
        states = ['punjab', 'haryana', 'up', 'maharashtra', 'gujarat', 'rajasthan', 
                  'mp', 'karnataka', 'tamil nadu', 'telangana', 'andhra', 'bengal', 
                  'odisha', 'bihar', 'assam']
        for state in states:
            if state in user_input_lower:
                info['location'] = state.title()
                break
        
        # Extract crop names
        crops = ['rice', 'wheat', 'cotton', 'maize', 'sugarcane', 'mustard', 
                 'gram', 'soybean', 'groundnut', 'tomato', 'potato', 'onion']
        for crop in crops:
            if crop in user_input_lower:
                info['crop'] = crop.title()
                break
        
        # Extract farm size
        acre_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:acre|acres)', user_input_lower)
        if acre_match:
            info['farm_size'] = f"{acre_match.group(1)} acres"
        
        hectare_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hectare|hectares|ha)', user_input_lower)
        if hectare_match:
            info['farm_size'] = f"{hectare_match.group(1)} hectares"
        
        return info
    
    def get_response(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        user_input_lower = user_input.lower().strip()
        
        # Extract contextual information
        extracted = self.extract_info(user_input)
        for key, value in extracted.items():
            if value:
                self.context[key] = value
        
        # Find matching topic
        matched_topic = None
        matched_category = None
        
        for category, data in self.knowledge_base.items():
            if category == "goodbye":
                continue
                
            keywords = data.get("keywords", [])
            for keyword in keywords:
                if keyword in user_input_lower:
                    matched_category = category
                    matched_topic = category
                    break
            
            if matched_category:
                break
        
        # Handle goodbye separately
        if not matched_category:
            goodbye_keywords = self.knowledge_base["goodbye"]["keywords"]
            for keyword in goodbye_keywords:
                if keyword in user_input_lower:
                    matched_category = "goodbye"
                    break
        
        # Generate response
        if matched_category:
            responses = self.knowledge_base[matched_category]["responses"]
            base_response = random.choice(responses)
            
            # Add specific information if available
            if matched_category == "crop_recommendation" and "info" in self.knowledge_base[matched_category]:
                if self.context.get('crop'):
                    crop_info = self.knowledge_base[matched_category]["info"].get(
                        self.context['crop'].lower(), ""
                    )
                    if crop_info:
                        base_response += f"\n\n📌 About {self.context['crop']}: {crop_info}"
            
            elif matched_category == "fertilizer" and "recommendations" in self.knowledge_base[matched_category]:
                if self.context.get('crop'):
                    fert_rec = self.knowledge_base[matched_category]["recommendations"].get(
                        self.context['crop'].lower(), ""
                    )
                    if fert_rec:
                        base_response += f"\n\n💡 Fertilizer recommendation for {self.context['crop']}: {fert_rec}"
            
            elif matched_category == "disease" and "common_diseases" in self.knowledge_base[matched_category]:
                # Check for specific disease mentions
                for disease_key, disease_info in self.knowledge_base[matched_category]["common_diseases"].items():
                    if disease_key.split('_')[0] in user_input_lower or disease_key.split('_')[1] in user_input_lower:
                        base_response += f"\n\n🔍 {disease_info}"
                        break
            
            self.last_topic = matched_category
            return base_response
        
        # Check for personal questions
        if any(word in user_input_lower for word in ["your name", "who are you", "what are you"]):
            return f"I am {self.name} v{self.version}, your intelligent agriculture assistant for the KrishiDisha project. I can help you with crop recommendations, fertilizer advice, disease management, and more!"
        
        if any(word in user_input_lower for word in ["what can you do", "help me", "services"]):
            return """I can assist you with:
🌱 Crop recommendations based on soil and climate
🧪 Fertilizer suggestions and NPK management
🐛 Disease identification and treatment
💧 Irrigation guidance
📅 Seasonal farming advice
🌾 Soil health tips
🥬 Organic farming practices
💰 Market prices and government schemes

Just ask me anything related to agriculture!"""
        
        # Default response for unmatched queries
        default_responses = [
            "I'm still learning about agriculture! Could you rephrase your question or ask about crops, fertilizers, diseases, or farming practices?",
            "That's an interesting question! For detailed agricultural advice, please ask about specific crops, soil types, or farming challenges.",
            "I specialize in agricultural guidance. Try asking about crop selection, fertilizer use, pest control, or irrigation methods.",
            "Could you provide more details? I can better assist with information about your location, soil type, or the specific crop you're interested in."
        ]
        
        return random.choice(default_responses)
    
    def chat(self):
        """Start interactive chat session"""
        print("=" * 70)
        print(f"  🌾 {self.name} - Agriculture LLM Chatbot v{self.version}")
        print("  Your Digital Farming Assistant for KrishiDisha Project")
        print("=" * 70)
        print(f"\n{self.name}: Namaste! I'm {self.name}, your agriculture assistant.")
        print("Ask me about crops, fertilizers, diseases, irrigation, or farming tips.")
        print("Type 'bye' or 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("👨‍🌾 You: ").strip()
                
                if not user_input:
                    continue
                
                response = self.get_response(user_input)
                print(f"\n🤖 {self.name}: {response}\n")
                
                # Check for exit
                if any(word in user_input.lower() for word in ["bye", "goodbye", "exit", "quit"]):
                    break
                    
            except KeyboardInterrupt:
                print(f"\n\n{self.name}: Goodbye! Wishing you a great harvest! 🌾")
                break
            except EOFError:
                break


def main():
    """Main function to run the chatbot"""
    bot = KrishiDishaBot()
    bot.chat()


if __name__ == "__main__":
    main()
