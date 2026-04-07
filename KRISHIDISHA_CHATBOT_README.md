# KrishiDisha Agriculture LLM Chatbot

## Overview
This is a specialized agriculture-focused chatbot for the **KrishiDisha** project, designed to assist farmers with crop recommendations, fertilizer advice, disease management, and general farming guidance.

## Features

### 🌱 Core Capabilities
- **Crop Recommendations**: Suggests suitable crops based on soil type, climate, and season
- **Fertilizer Guidance**: Provides NPK recommendations for different crops
- **Disease Management**: Identifies common crop diseases and suggests treatments
- **Irrigation Advice**: Water management tips for different crops
- **Seasonal Farming**: Information about Kharif, Rabi, and Zaid seasons
- **Soil Health**: Soil type analysis and improvement suggestions
- **Organic Farming**: Natural pest control and organic practices
- **Market Information**: MSP, e-NAM, and market access guidance
- **Government Schemes**: PM-KISAN, PMFBY, KCC, and other schemes

### 🤖 Intelligent Features
- Context-aware conversations
- Information extraction (location, crop names, farm size)
- Topic-specific detailed responses
- Multi-language greeting support (including "Namaste")
- Emoji-enhanced user interface

## Usage

### Running the Chatbot
```bash
python3 krishidisha_bot.py
```

### Example Conversation
```
👨‍🌾 You: hello
🤖 KrishiDisha: Namaste! Welcome to KrishiDisha...

👨‍🌾 You: what crops can I grow in Punjab?
🤖 KrishiDisha: I can help recommend suitable crops!...

👨‍🌾 You: tell me about fertilizer for wheat
🤖 KrishiDisha: Fertilizer selection depends on soil nutrient levels...
💡 Fertilizer recommendation for Wheat: Use 120-150 kg N...

👨‍🌾 You: how to prevent rice blast disease
🤖 KrishiDisha: Plant diseases can be identified by symptoms...
🔍 Caused by fungus Magnaporthe oryzae...
```

## Knowledge Base Categories

1. **Greeting**: Welcome messages in multiple styles
2. **Crop Recommendation**: Crop-specific growing information
3. **Fertilizer**: NPK recommendations for major crops
4. **Disease**: Common diseases and treatment methods
5. **Irrigation**: Water management strategies
6. **Season**: Indian cropping seasons information
7. **Soil**: Soil types and characteristics
8. **Organic**: Organic farming practices
9. **Market**: Market prices and selling strategies
10. **Government Schemes**: Subsidies and support programs

## Supported Crops
- Rice, Wheat, Cotton, Maize
- Sugarcane, Mustard, Gram
- Soybean, Groundnut, Tomato
- Potato, Onion, and more

## Integration with KrishiDisha Project

This chatbot complements the existing KrishiDisha system which includes:
- Crop recommendation ML models
- Disease detection using CNN
- Fertilizer prediction systems
- Farmer management portal
- Admin dashboard

## Future Enhancements

Potential improvements for production deployment:
- [ ] Integration with actual ML models for real-time predictions
- [ ] Database connectivity for personalized farmer profiles
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Voice interface integration
- [ ] SMS/WhatsApp bot version
- [ ] Weather API integration
- [ ] Real-time market price feeds
- [ ] Image-based disease detection
- [ ] Location-specific recommendations using GPS

## Files

- `krishidisha_bot.py` - Main chatbot implementation
- `chatbot.py` - Original simple chatbot (reference)
- `app.py` - Flask web application for KrishiDisha
- `requirements.txt` - Python dependencies

## Authors

Developed for the KrishiDisha Project by:
- Abhishek Chourasia (EN23CS3T1013)
- Goutam Mandloi (EN23CS3T1016)
- Shivpratap Singh Panwar (EN23CS3T1017)

## License

Part of the KrishiDisha agricultural assistance project.

---

**Jai Kisan! 🌾**
