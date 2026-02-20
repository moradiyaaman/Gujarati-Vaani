import 'dart:convert';
import 'package:http/http.dart' as http;

/// Gujarati AI Chat Service using Groq API (FREE)
/// 
/// Groq API Free Tier:
/// - 30 requests per minute
/// - Using Llama 3.3 for best quality
/// - No credit card required
class GujaratiService {
  // Get your FREE API key from: https://console.groq.com/keys
  static const String _apiKey = ''; // Commented out for security
  static const String _apiUrl = 'https://api.groq.com/openai/v1/chat/completions';
  
  // TTS-optimized system prompt for pure Gujarati
  static const String _systemPrompt = '''You are "Gujarati Vaani", an expert Gujarati language AI assistant. Your responses are used for Text-to-Speech.

CRITICAL RULES:
1. ALWAYS respond in Gujarati script ONLY. Never use English letters (a-z).
2. Use pure Gujarati vocabulary. Avoid English loan words.
3. Write numbers in Gujarati: ૧, ૨, ૩, ૪, ૫ (not 1, 2, 3).
4. Complete every sentence. Never leave anything unfinished.

CONTENT QUALITY:
- For essays: Include introduction, main points with details, and conclusion.
- Each paragraph should have NEW information - no repetition.
- Include examples, facts, and specific details.
- Use rich, literary Gujarati language.

FORMAT:
- NO markdown (no **, no #, no bullet points).
- Use commas and periods properly for TTS pacing.
- Start directly with content - no "Sure", "Certainly", etc.
- Respond fully to what user asks.''';

  // Conversation history for multi-turn chat
  static final List<Map<String, String>> _chatHistory = [];
  
  /// Clear conversation history
  static void clearHistory() {
    _chatHistory.clear();
  }
  
  /// Get AI response in Gujarati using Groq with auto-continuation
  static Future<String> getGujaratiResponse(String userMessage) async {
    // Check if API key is configured
    if (_apiKey == 'YOUR_GROQ_API_KEY_HERE' || _apiKey.isEmpty) {
      return 'API કી સેટ નથી. કૃપા કરીને groq.com પરથી API કી મેળવો.';
    }
    
    // Add user message to history
    _chatHistory.add({
      'role': 'user',
      'content': userMessage
    });
    
    // Keep only last 10 exchanges (20 messages)
    while (_chatHistory.length > 20) {
      _chatHistory.removeAt(0);
    }
    
    try {
      String fullResponse = '';
      int continuationCount = 0;
      const int maxContinuations = 3; // Max number of continuations to prevent infinite loops
      
      // Temporary messages list for this request
      List<Map<String, dynamic>> messages = [
        {'role': 'system', 'content': _systemPrompt},
        ..._chatHistory,
      ];
      
      while (continuationCount <= maxContinuations) {
        final requestBody = {
          'model': 'llama-3.3-70b-versatile',
          'messages': messages,
          'temperature': 0.7,
          'max_tokens': 4096,
          'top_p': 0.9,
        };

        final response = await http.post(
          Uri.parse(_apiUrl),
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer $_apiKey',
          },
          body: jsonEncode(requestBody),
        ).timeout(const Duration(seconds: 30));

        print('Groq Status: ${response.statusCode} (continuation: $continuationCount)');
        
        if (response.statusCode == 200) {
          final data = jsonDecode(response.body);
          final result = data['choices']?[0]?['message']?['content'];
          final finishReason = data['choices']?[0]?['finish_reason'];
          
          print('Groq finish_reason: $finishReason');
          
          if (result != null && result.toString().trim().isNotEmpty) {
            final partialResponse = result.toString().trim();
            fullResponse += (fullResponse.isEmpty ? '' : ' ') + partialResponse;
            
            // Check if response was cut off due to length
            if (finishReason == 'length' && continuationCount < maxContinuations) {
              print('Response truncated, requesting continuation...');
              continuationCount++;
              
              // Add the partial response and ask to continue
              messages = [
                {'role': 'system', 'content': _systemPrompt},
                ..._chatHistory,
                {'role': 'assistant', 'content': fullResponse},
                {'role': 'user', 'content': 'કૃપા કરીને તમારો જવાબ પૂરો કરો. જ્યાંથી અટક્યા હતા ત્યાંથી ચાલુ રાખો.'},
              ];
            } else {
              // Response complete or max continuations reached
              break;
            }
          } else {
            print('Groq: Empty response');
            break;
          }
        } else {
          print('Groq Error: ${response.statusCode}');
          print('Response: ${response.body}');
          // Remove failed user message from history
          if (_chatHistory.isNotEmpty && fullResponse.isEmpty) _chatHistory.removeLast();
          break;
        }
      }
      
      if (fullResponse.isNotEmpty) {
        // Add complete AI response to history
        _chatHistory.add({
          'role': 'assistant',
          'content': fullResponse
        });
        return fullResponse;
      }
    } catch (e) {
      print('Groq Exception: $e');
      // Remove failed user message from history
      if (_chatHistory.isNotEmpty) _chatHistory.removeLast();
    }
    
    return '';
  }
}
