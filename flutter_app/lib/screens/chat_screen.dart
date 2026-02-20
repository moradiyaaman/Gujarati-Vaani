import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:audioplayers/audioplayers.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:speech_to_text/speech_recognition_result.dart';
import '../services/tts_service.dart';
import '../services/gujarati_service.dart';

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  String? audioPath;
  bool isGeneratingAudio;

  ChatMessage({
    required this.text,
    required this.isUser,
    DateTime? timestamp,
    this.audioPath,
    this.isGeneratingAudio = false,
  }) : timestamp = timestamp ?? DateTime.now();
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final AudioPlayer _audioPlayer = AudioPlayer();
  final List<ChatMessage> _messages = [];
  bool _isTyping = false;
  int? _playingIndex;
  bool _autoPlayResponses = true;

  // Speech-to-text
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _isListening = false;
  bool _speechEnabled = false;
  String _lastWords = '';

  // Free AI APIs
  static const String _hfApiUrl =
      'https://api-inference.huggingface.co/models/google/gemma-2-2b-it';
  static const String _hfApiUrl2 =
      'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3';

  @override
  void initState() {
    super.initState();
    _setupAudioPlayer();
    _initSpeech();
    // Welcome message
    _messages.add(ChatMessage(
      text:
          'નમસ્તે! હું "ગુજરાતી વાણી" AI સહાયક છું।\n\n✨ મને ગુજરાતીમાં કંઈપણ પૂછો\n🎙️ બોલવા માટે માઇક દબાવો\n🔊 જવાબ સાંભળવા Listen દબાવો\n🗑️ નવી વાતચીત શરૂ કરવા Clear દબાવો\n\nહું આ વિષયોમાં મદદ કરી શકું:\n• ગુજરાતનો ઈતિહાસ અને સંસ્કૃતિ\n• ખાણીપીણી અને વાનગીઓ\n• તહેવારો અને પરંપરાઓ\n• પ્રવાસન સ્થળો\n• સામાન્ય જ્ઞાન',
      isUser: false,
    ));
  }

  /// Initialize speech recognition
  void _initSpeech() async {
    _speechEnabled = await _speech.initialize(
      onStatus: (status) {
        print('Speech status: $status');
        if (status == 'done' || status == 'notListening') {
          if (mounted) setState(() => _isListening = false);
        }
      },
      onError: (error) {
        print('Speech error: $error');
        if (mounted) setState(() => _isListening = false);
      },
    );
    setState(() {});
  }

  /// Start or stop listening
  void _toggleListening() async {
    if (_isListening) {
      await _speech.stop();
      setState(() => _isListening = false);
    } else {
      if (_speechEnabled) {
        setState(() => _isListening = true);
        await _speech.listen(
          onResult: _onSpeechResult,
          localeId: 'gu_IN', // Gujarati locale
          listenFor: const Duration(seconds: 30),
          pauseFor: const Duration(seconds: 3),
          partialResults: true,
          cancelOnError: true,
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('માઇક્રોફોન ઉપલબ્ધ નથી. કૃપા કરીને પરવાનગી આપો.'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  /// Handle speech recognition results
  void _onSpeechResult(SpeechRecognitionResult result) {
    setState(() {
      _lastWords = result.recognizedWords;
      _messageController.text = _lastWords;
    });
    
    // Auto-send when speech recognition is final
    if (result.finalResult && _lastWords.isNotEmpty) {
      Future.delayed(const Duration(milliseconds: 500), () {
        if (mounted && _messageController.text.isNotEmpty) {
          _sendMessage();
        }
      });
    }
  }

  void _setupAudioPlayer() {
    _audioPlayer.onPlayerComplete.listen((_) {
      if (mounted) setState(() => _playingIndex = null);
    });
  }

  @override
  void dispose() {
    _speech.stop(); // Stop any speech recognition
    _messageController.dispose();
    _scrollController.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  /// Clear chat history and start fresh
  void _clearChat() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('વાતચીત ભૂંસો?'),
        content: const Text('શું તમે ખરેખર બધી વાતચીત ભૂંસવા માંગો છો?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('રદ કરો'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              setState(() {
                _messages.clear();
                _playingIndex = null;
                GujaratiService.clearHistory();
                // Add welcome message again
                _messages.add(ChatMessage(
                  text: 'વાતચીત ભૂંસી નાખી! નવો પ્રશ્ન પૂછો। 😊',
                  isUser: false,
                ));
              });
            },
            child: const Text('ભૂંસો', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _sendMessage() async {
    final text = _messageController.text.trim();
    if (text.isEmpty) return;

    _messageController.clear();

    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isTyping = true;
    });
    _scrollToBottom();

    try {
      final response = await _getAIResponse(text);
      
      if (!mounted) return;
      
      setState(() {
        _messages.add(ChatMessage(text: response, isUser: false));
        _isTyping = false;
      });
      _scrollToBottom();

      // Auto-play the response if enabled
      if (_autoPlayResponses) {
        Future.delayed(const Duration(milliseconds: 300), () {
          if (mounted) _playMessageAudio(_messages.length - 1);
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messages.add(ChatMessage(
          text: 'માફ કરશો, કંઈક ખોટું થયું. ફરી પ્રયાસ કરો.',
          isUser: false,
        ));
        _isTyping = false;
      });
    }
  }

  Future<String> _getAIResponse(String userMessage) async {
    // Try Groq first (best for Gujarati)
    try {
      final groqResponse = await GujaratiService.getGujaratiResponse(userMessage);
      if (groqResponse.isNotEmpty) {
        return groqResponse;
      }
    } catch (e) {
      print('Groq error: $e');
    }

    // Fallback to HuggingFace
    try {
      final response = await http.post(
        Uri.parse(_hfApiUrl),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'inputs':
              'You are a helpful AI assistant. Respond in Gujarati. User: $userMessage',
          'parameters': {
            'max_new_tokens': 200,
            'temperature': 0.7,
          },
        }),
      ).timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data is List && data.isNotEmpty) {
          String text = data[0]['generated_text'] ?? '';
          // Extract response after User message
          if (text.contains('User:')) {
            text = text.split('User:').last;
            if (text.contains('\n')) {
              text = text.split('\n').skip(1).join('\n').trim();
            }
          }
          if (text.isNotEmpty) return text;
        }
      }
    } catch (e) {
      // Continue to next fallback
    }

    // Smart fallback response in Gujarati
    final fallbacks = [
      'કૃપા કરીને ઉપરના કોઈ વિષય વિશે ગુજરાતીમાં પૂછો!',
      'તમારો પ્રશ્ન સારો છે! હું ગુજરાતી ભાષામાં ઘણા વિષયો વિશે માહિતી આપી શકું છું:\n\n• ગુજરાતનો ઈતિહાસ અને સંસ્કૃતિ\n• ખાણીપીણી અને વાનગીઓ\n• તહેવારો (નવરાત્રી, ઉત્તરાયણ)\n• પ્રવાસન (ગીર, કચ્છ, સ્ટેચ્યૂ)\n• વિજ્ઞાન અને ટેકનોલોજી\n• સાહિત્ય અને ભાષા\n\nકૃપા કરીને ઉપરના કોઈ વિષય વિશે ગુજરાતીમાં પૂછો!',
      'હું તમને ગુજરાતી ભાષામાં મદદ કરવા તૈયાર છું. કૃપા કરી તમારો પ્રશ્ન વિગતવાર જણાવો.',
    ];
    return fallbacks[DateTime.now().millisecond % fallbacks.length];
  }

  Future<void> _playMessageAudio(int index) async {
    if (index >= _messages.length) return;
    final message = _messages[index];
    if (message.isUser) return;

    // If already playing this message, stop
    if (_playingIndex == index) {
      await _audioPlayer.stop();
      setState(() => _playingIndex = null);
      return;
    }

    // If audio already exists, play it
    if (message.audioPath != null) {
      setState(() => _playingIndex = index);
      await _audioPlayer.play(DeviceFileSource(message.audioPath!));
      return;
    }

    // Generate audio
    setState(() {
      _messages[index].isGeneratingAudio = true;
    });

    try {
      final audioPath = await TTSService.synthesize(message.text);
      if (audioPath != null && mounted) {
        setState(() {
          _messages[index].audioPath = audioPath;
          _messages[index].isGeneratingAudio = false;
          _playingIndex = index;
        });
        await _audioPlayer.play(DeviceFileSource(audioPath));
      } else {
        setState(() {
          _messages[index].isGeneratingAudio = false;
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('ઓડિયો બનાવવામાં ભૂલ થઈ'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _messages[index].isGeneratingAudio = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Messages list
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
              itemCount: _messages.length + (_isTyping ? 1 : 0),
              itemBuilder: (context, index) {
                if (index == _messages.length && _isTyping) {
                  return _buildTypingIndicator();
                }
                return _buildMessageBubble(index);
              },
            ),
          ),

          // Input area
          Container(
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.08),
                  blurRadius: 8,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            padding: const EdgeInsets.fromLTRB(8, 8, 8, 12),
            child: Column(
              children: [
                // Auto-play toggle
                Padding(
                  padding: const EdgeInsets.only(bottom: 8, left: 4, right: 4),
                  child: Row(
                    children: [
                      InkWell(
                        onTap: () => setState(() => _autoPlayResponses = !_autoPlayResponses),
                        borderRadius: BorderRadius.circular(20),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: _autoPlayResponses ? Colors.orange[100] : Colors.grey[100],
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: _autoPlayResponses ? Colors.orange : Colors.grey[300]!,
                            ),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                _autoPlayResponses ? Icons.volume_up : Icons.volume_off,
                                size: 16,
                                color: _autoPlayResponses ? Colors.orange[700] : Colors.grey[600],
                              ),
                              const SizedBox(width: 6),
                              Text(
                                'Auto Play 🔊',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: _autoPlayResponses ? Colors.orange[700] : Colors.grey[600],
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      // Clear chat button
                      InkWell(
                        onTap: _clearChat,
                        borderRadius: BorderRadius.circular(20),
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: Colors.red[50],
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(color: Colors.red[200]!),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.delete_outline, size: 16, color: Colors.red[700]),
                              const SizedBox(width: 6),
                              Text(
                                'Clear',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.red[700],
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                // Input row
                Row(
                  children: [
                    // Microphone button
                    Container(
                      decoration: BoxDecoration(
                        color: _isListening ? Colors.red : Colors.grey[200],
                        shape: BoxShape.circle,
                      ),
                      child: IconButton(
                        onPressed: _toggleListening,
                        icon: Icon(_isListening ? Icons.mic : Icons.mic_none),
                        color: _isListening ? Colors.white : Colors.grey[700],
                        iconSize: 22,
                        tooltip: 'બોલો',
                      ),
                    ),
                    const SizedBox(width: 8),
                    // Text input
                    Expanded(
                      child: TextField(
                        controller: _messageController,
                        decoration: InputDecoration(
                          hintText: _isListening 
                              ? 'બોલો... 🎤' 
                              : 'ગુજરાતીમાં લખો... (Type in Gujarati...)',
                          hintStyle: TextStyle(
                            color: _isListening ? Colors.red[400] : Colors.grey[400], 
                            fontSize: 14,
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide(color: _isListening ? Colors.red : Colors.grey[300]!),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide(color: _isListening ? Colors.red : Colors.orange, width: 2),
                          ),
                          filled: true,
                          fillColor: _isListening ? Colors.red[50] : Colors.grey[50],
                          contentPadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
                        ),
                        style: const TextStyle(fontSize: 16),
                        maxLines: 3,
                        minLines: 1,
                        textInputAction: TextInputAction.send,
                        onSubmitted: (_) => _sendMessage(),
                      ),
                    ),
                    const SizedBox(width: 8),
                    // Send button
                    Container(
                      decoration: const BoxDecoration(
                        color: Colors.orange,
                        shape: BoxShape.circle,
                      ),
                      child: IconButton(
                        onPressed: _isTyping ? null : () => _sendMessage(),
                        icon: const Icon(Icons.send_rounded),
                        color: Colors.white,
                        iconSize: 22,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTypingIndicator() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8, right: 60),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.grey[100],
          borderRadius: BorderRadius.circular(18),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildDot(0),
            const SizedBox(width: 4),
            _buildDot(1),
            const SizedBox(width: 4),
            _buildDot(2),
          ],
        ),
      ),
    );
  }

  Widget _buildDot(int index) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: Duration(milliseconds: 600 + (index * 200)),
      builder: (context, value, child) {
        return Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: Colors.grey[400]!.withOpacity(0.5 + (value * 0.5)),
            shape: BoxShape.circle,
          ),
        );
      },
    );
  }

  Widget _buildMessageBubble(int index) {
    final message = _messages[index];
    final isUser = message.isUser;
    final isPlaying = _playingIndex == index;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.only(
          bottom: 10,
          left: isUser ? 50 : 0,
          right: isUser ? 0 : 50,
        ),
        child: Column(
          crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: isUser ? Colors.orange : Colors.grey[100],
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(18),
                  topRight: const Radius.circular(18),
                  bottomLeft: Radius.circular(isUser ? 18 : 4),
                  bottomRight: Radius.circular(isUser ? 4 : 18),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.06),
                    blurRadius: 6,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Text(
                message.text,
                style: TextStyle(
                  fontSize: 15,
                  color: isUser ? Colors.white : Colors.black87,
                  height: 1.4,
                ),
              ),
            ),
            const SizedBox(height: 4),
            // Listen button for AI responses
            if (!isUser)
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (message.isGeneratingAudio)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.orange[50],
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.orange[700],
                            ),
                          ),
                          const SizedBox(width: 6),
                          Text('Generating...',
                              style: TextStyle(fontSize: 11, color: Colors.orange[700])),
                        ],
                      ),
                    )
                  else
                    InkWell(
                      onTap: () => _playMessageAudio(index),
                      borderRadius: BorderRadius.circular(12),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: isPlaying ? Colors.orange[100] : Colors.orange[50],
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: isPlaying ? Colors.orange : Colors.orange[200]!,
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              isPlaying ? Icons.stop_rounded : Icons.volume_up_rounded,
                              size: 16,
                              color: Colors.orange[700],
                            ),
                            const SizedBox(width: 4),
                            Text(
                              isPlaying ? 'Stop' : 'Listen 🔊',
                              style: TextStyle(
                                fontSize: 11,
                                color: Colors.orange[700],
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  const SizedBox(width: 8),
                  Text(
                    '${message.timestamp.hour}:${message.timestamp.minute.toString().padLeft(2, '0')}',
                    style: TextStyle(fontSize: 10, color: Colors.grey[400]),
                  ),
                ],
              ),
            if (isUser)
              Padding(
                padding: const EdgeInsets.only(right: 4),
                child: Text(
                  '${message.timestamp.hour}:${message.timestamp.minute.toString().padLeft(2, '0')}',
                  style: TextStyle(fontSize: 10, color: Colors.grey[400]),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
