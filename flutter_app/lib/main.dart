import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'dart:async';
import 'package:path_provider/path_provider.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:share_plus/share_plus.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const GujaratiVaaniApp());
}

class GujaratiVaaniApp extends StatelessWidget {
  const GujaratiVaaniApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gujarati Vaani',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.orange,
        useMaterial3: true,
      ),
      home: const TTSHomePage(),
    );
  }
}

class TTSHomePage extends StatefulWidget {
  const TTSHomePage({super.key});

  @override
  State<TTSHomePage> createState() => _TTSHomePageState();
}

class _TTSHomePageState extends State<TTSHomePage> {
  final TextEditingController _textController = TextEditingController();
  final AudioPlayer _audioPlayer = AudioPlayer();
  
  static const String apiUrl = 'https://moradiyaaman-gujarati-vaani-tts.hf.space/synthesize';
  
  bool _isLoading = false;
  double _playbackSpeed = 1.0;
  String? _audioPath;
  bool _isPlaying = false;
  
  // Progress tracking
  String _statusMessage = '';
  int _estimatedSeconds = 0;
  int _elapsedSeconds = 0;
  Timer? _progressTimer;
  
  // Audio player state
  Duration _audioDuration = Duration.zero;
  Duration _audioPosition = Duration.zero;
  bool _audioGenerated = false;

  @override
  void initState() {
    super.initState();
    _setupAudioPlayer();
  }

  void _setupAudioPlayer() {
    _audioPlayer.onDurationChanged.listen((duration) {
      if (mounted) {
        setState(() {
          _audioDuration = duration;
        });
      }
    });

    _audioPlayer.onPositionChanged.listen((position) {
      if (mounted) {
        setState(() {
          _audioPosition = position;
        });
      }
    });

    _audioPlayer.onPlayerComplete.listen((event) {
      if (mounted) {
        setState(() {
          _isPlaying = false;
          _audioPosition = Duration.zero;
        });
      }
    });
  }

  @override
  void dispose() {
    _textController.dispose();
    _audioPlayer.dispose();
    _progressTimer?.cancel();
    super.dispose();
  }

  // Check internet connection
  Future<bool> _checkInternetConnection() async {
    try {
      final result = await InternetAddress.lookup('google.com')
          .timeout(const Duration(seconds: 5));
      return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
    } on SocketException catch (_) {
      return false;
    } on TimeoutException catch (_) {
      return false;
    }
  }

  // Calculate estimated time based on text length
  int _calculateEstimatedTime(String text) {
    int charCount = text.length;
    // For chunked processing: ~25 seconds per 500 characters (based on real testing)
    int numChunks = (charCount / 500).ceil();
    if (numChunks < 1) numChunks = 1;
    int estimatedSeconds = numChunks * 25; // 25 seconds per chunk is more realistic
    return estimatedSeconds.clamp(10, 1200);
  }

  // Validate if text contains Gujarati characters
  bool _isGujaratiText(String text) {
    // Gujarati Unicode range: U+0A80 to U+0AFF
    // Also check for digits in Gujarati: U+0AE6 to U+0AEF
    final gujaratiPattern = RegExp(r'[\u0A80-\u0AFF]');
    return gujaratiPattern.hasMatch(text);
  }

  // Detect predominant language in text
  String _detectLanguage(String text) {
    final gujaratiChars = RegExp(r'[\u0A80-\u0AFF]').allMatches(text).length;
    final hindiChars = RegExp(r'[\u0900-\u097F]').allMatches(text).length;
    final englishChars = RegExp(r'[a-zA-Z]').allMatches(text).length;
    final otherIndianChars = RegExp(r'[\u0980-\u09FF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]').allMatches(text).length;
    
    // Remove punctuation and spaces for better analysis
    final cleanText = text.replaceAll(RegExp(r'[\s\p{P}\d]'), '');
    final totalChars = cleanText.length;
    
    if (totalChars == 0) return 'unknown';
    
    // If more than 80% Gujarati characters, consider it Gujarati
    if (gujaratiChars > 0 && (gujaratiChars / totalChars) > 0.8) {
      return 'gujarati';
    }
    
    // Detect other languages
    if (hindiChars > gujaratiChars && hindiChars > englishChars) {
      return 'hindi';
    }
    if (englishChars > gujaratiChars && englishChars > hindiChars) {
      return 'english';
    }
    if (otherIndianChars > gujaratiChars) {
      return 'other_indian';
    }
    
    // If some Gujarati but not dominant
    if (gujaratiChars > 0) {
      return 'mixed';
    }
    
    return 'unknown';
  }

  // Start progress timer
  void _startProgressTimer() {
    _elapsedSeconds = 0;
    _progressTimer?.cancel();
    _progressTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (mounted) {
        setState(() {
          _elapsedSeconds++;
        });
      }
    });
  }

  // Stop progress timer
  void _stopProgressTimer() {
    _progressTimer?.cancel();
    _progressTimer = null;
  }

  // Split text into chunks for processing
  List<String> _splitTextIntoChunks(String text, int maxChunkSize) {
    List<String> chunks = [];
    
    // First, protect decimal numbers by temporarily replacing them
    // Match decimal numbers like ૪.૫ or 4.5 and protect them
    RegExp decimalPattern = RegExp(r'(\d+\.\d+|[૦-૯]+\.[૦-૯]+)');
    Map<String, String> decimalMap = {};
    int decimalIndex = 0;
    
    String protectedText = text.replaceAllMapped(decimalPattern, (match) {
      String placeholder = '<<<DECIMAL$decimalIndex>>>';
      decimalMap[placeholder] = match.group(0)!;
      decimalIndex++;
      return placeholder;
    });
    
    // Split by Gujarati sentence endings:
    // - ।  (Gujarati danda - primary sentence delimiter)
    // - .  followed by space or end (English period as sentence end)
    // - !  followed by space or end
    // - ?  followed by space or end
    RegExp sentenceEnd = RegExp(r'।|\.(?=\s|$)|!(?=\s|$)|\?(?=\s|$)');
    
    List<String> parts = protectedText.split(sentenceEnd);
    List<Match> delimiters = sentenceEnd.allMatches(protectedText).toList();
    
    String currentChunk = '';
    for (int i = 0; i < parts.length; i++) {
      String sentence = parts[i].trim();
      if (sentence.isEmpty) continue;
      
      // Restore decimal numbers in this sentence
      decimalMap.forEach((placeholder, original) {
        sentence = sentence.replaceAll(placeholder, original);
      });
      
      // Add back the appropriate punctuation
      String delimiter = (i < delimiters.length) ? delimiters[i].group(0)! : '';
      // Use Gujarati danda for consistency
      String sentenceWithPunct = sentence + (delimiter.isNotEmpty ? delimiter : '।') + ' ';
      
      if ((currentChunk + sentenceWithPunct).length > maxChunkSize && currentChunk.isNotEmpty) {
        chunks.add(currentChunk.trim());
        currentChunk = sentenceWithPunct;
      } else {
        currentChunk += sentenceWithPunct;
      }
    }
    
    if (currentChunk.trim().isNotEmpty) {
      chunks.add(currentChunk.trim());
    }
    
    // If no chunks created (no sentence endings), split by character limit
    if (chunks.isEmpty && text.isNotEmpty) {
      for (int i = 0; i < text.length; i += maxChunkSize) {
        int end = (i + maxChunkSize < text.length) ? i + maxChunkSize : text.length;
        chunks.add(text.substring(i, end));
      }
    }
    
    return chunks;
  }

  Future<void> _generateAudio() async {
    final text = _textController.text.trim();
    
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter some Gujarati text'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    // Validate Gujarati language
    final detectedLanguage = _detectLanguage(text);
    
    if (detectedLanguage != 'gujarati' && detectedLanguage != 'mixed') {
      String errorMessage;
      IconData errorIcon;
      
      switch (detectedLanguage) {
        case 'hindi':
          errorMessage = 'કૃપા કરીને ગુજરાતી ટેક્સ્ટ દાખલ કરો। આ હિન્દી ભાષા છે.\n(Please enter Gujarati text. This is Hindi language.)';
          errorIcon = Icons.translate;
          break;
        case 'english':
          errorMessage = 'કૃપા કરીને ગુજરાતી ટેક્સ્ટ દાખલ કરો। આ અંગ્રેજી ભાષા છે.\n(Please enter Gujarati text. This is English language.)';
          errorIcon = Icons.translate;
          break;
        case 'other_indian':
          errorMessage = 'કૃપા કરીને ગુજરાતી ટેક્સ્ટ દાખલ કરો। આ અન્ય ભારતીય ભાષા છે.\n(Please enter Gujarati text. This is another Indian language.)';
          errorIcon = Icons.translate;
          break;
        default:
          errorMessage = 'કૃપા કરીને ગુજરાતી ટેક્સ્ટ દાખલ કરો। આ ટેક્સ્ટ ગુજરાતીમાં નથી.\n(Please enter Gujarati text. This text is not in Gujarati.)';
          errorIcon = Icons.warning;
      }
      
      showDialog(
        context: context,
        builder: (BuildContext dialogContext) {
          return AlertDialog(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            title: Row(
              children: [
                Icon(errorIcon, color: Colors.red, size: 28),
                SizedBox(width: 10),
                Text(
                  'Wrong Language',
                  style: TextStyle(
                    color: Colors.red,
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ],
            ),
            content: Text(
              errorMessage,
              style: TextStyle(fontSize: 15, height: 1.4),
            ),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(dialogContext).pop();
                  _textController.clear();
                  setState(() {});
                },
                style: TextButton.styleFrom(foregroundColor: Colors.red),
                child: Text('Clear Text', style: TextStyle(fontSize: 15)),
              ),
              ElevatedButton(
                onPressed: () {
                  Navigator.of(dialogContext).pop();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: Text('OK', style: TextStyle(fontSize: 15)),
              ),
            ],
          );
        },
      );
      return;
    }

    // Reset audio state
    setState(() {
      _audioGenerated = false;
      _audioPath = null;
      _isPlaying = false;
      _audioPosition = Duration.zero;
      _audioDuration = Duration.zero;
    });

    // Check internet connection first
    setState(() {
      _statusMessage = 'Connecting...';
      _isLoading = true;
    });

    bool hasInternet = await _checkInternetConnection();
    if (!hasInternet) {
      setState(() {
        _isLoading = false;
        _statusMessage = '';
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Row(
            children: [
              Icon(Icons.wifi_off, color: Colors.white),
              SizedBox(width: 8),
              Expanded(child: Text('No internet connection. Please check your network.')),
            ],
          ),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 4),
        ),
      );
      return;
    }

    // For large texts, split into chunks
    const int maxChunkSize = 500; // characters per chunk
    List<String> chunks;
    
    if (text.length > maxChunkSize) {
      chunks = _splitTextIntoChunks(text, maxChunkSize);
    } else {
      chunks = [text];
    }
    
    // Calculate estimated time based on chunks
    _estimatedSeconds = chunks.length * 10; // ~10 seconds per chunk
    
    _startProgressTimer();
    
    List<List<int>> audioBytes = [];
    
    for (int i = 0; i < chunks.length; i++) {
      setState(() {
        _statusMessage = 'Processing ${i + 1}/${chunks.length}...';
      });
      
      try {
        final response = await http.post(
          Uri.parse(apiUrl),
          headers: {
            'Content-Type': 'application/json; charset=utf-8',
          },
          body: json.encode({
            'text': chunks[i],
            'speed': 1.0,
          }),
        ).timeout(const Duration(seconds: 120));

        if (response.statusCode == 200) {
          audioBytes.add(response.bodyBytes);
        } else if (response.statusCode == 503) {
          // Server starting, wait and retry this chunk
          setState(() {
            _statusMessage = 'Server starting, retrying...';
          });
          await Future.delayed(const Duration(seconds: 10));
          i--; // Retry same chunk
          continue;
        } else {
          throw Exception('Server error: ${response.statusCode}');
        }
      } on TimeoutException {
        // Retry once on timeout
        setState(() {
          _statusMessage = 'Retrying ${i + 1}/${chunks.length}...';
        });
        try {
          final response = await http.post(
            Uri.parse(apiUrl),
            headers: {
              'Content-Type': 'application/json; charset=utf-8',
            },
            body: json.encode({
              'text': chunks[i],
              'speed': 1.0,
            }),
          ).timeout(const Duration(seconds: 180));
          
          if (response.statusCode == 200) {
            audioBytes.add(response.bodyBytes);
          } else {
            throw Exception('Failed after retry');
          }
        } catch (e) {
          _stopProgressTimer();
          setState(() {
            _isLoading = false;
            _statusMessage = '';
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Failed at part ${i + 1}. Try with shorter text.'),
              backgroundColor: Colors.red,
            ),
          );
          return;
        }
      } catch (e) {
        _stopProgressTimer();
        setState(() {
          _isLoading = false;
          _statusMessage = '';
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error at part ${i + 1}: ${e.toString().split(':').last}'),
            backgroundColor: Colors.red,
          ),
        );
        return;
      }
    }
    
    _stopProgressTimer();
    
    // Combine all audio chunks
    if (audioBytes.isNotEmpty) {
      setState(() {
        _statusMessage = 'Combining audio...';
      });
      
      try {
        final directory = await getTemporaryDirectory();
        final filePath = '${directory.path}/gujarati_tts_${DateTime.now().millisecondsSinceEpoch}.wav';
        
        // Simple concatenation - combine WAV data
        // For WAV files, we need to handle headers properly
        List<int> combinedAudio = [];
        
        if (audioBytes.length == 1) {
          // Single chunk, use as-is
          combinedAudio = audioBytes[0];
        } else {
          // Multiple chunks - combine raw audio data
          // First file: keep full header (44 bytes for WAV)
          combinedAudio.addAll(audioBytes[0]);
          
          // Subsequent files: skip WAV header (44 bytes)
          for (int i = 1; i < audioBytes.length; i++) {
            if (audioBytes[i].length > 44) {
              combinedAudio.addAll(audioBytes[i].sublist(44));
            }
          }
          
          // Update WAV header with new file size
          int dataSize = combinedAudio.length - 44;
          int fileSize = combinedAudio.length - 8;
          
          // Update RIFF chunk size (bytes 4-7)
          combinedAudio[4] = fileSize & 0xFF;
          combinedAudio[5] = (fileSize >> 8) & 0xFF;
          combinedAudio[6] = (fileSize >> 16) & 0xFF;
          combinedAudio[7] = (fileSize >> 24) & 0xFF;
          
          // Update data chunk size (bytes 40-43)
          combinedAudio[40] = dataSize & 0xFF;
          combinedAudio[41] = (dataSize >> 8) & 0xFF;
          combinedAudio[42] = (dataSize >> 16) & 0xFF;
          combinedAudio[43] = (dataSize >> 24) & 0xFF;
        }
        
        final file = File(filePath);
        await file.writeAsBytes(combinedAudio);

        setState(() {
          _audioPath = filePath;
          _isLoading = false;
          _statusMessage = '';
          _audioGenerated = true;
        });

        // Auto-play the generated audio
        await _playAudio();
        
      } catch (e) {
        setState(() {
          _isLoading = false;
          _statusMessage = '';
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Error saving audio file'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _playAudio() async {
    if (_audioPath == null) return;

    try {
      if (_isPlaying) {
        await _audioPlayer.pause();
        setState(() {
          _isPlaying = false;
        });
      } else {
        await _audioPlayer.setPlaybackRate(_playbackSpeed);
        await _audioPlayer.play(DeviceFileSource(_audioPath!));
        setState(() {
          _isPlaying = true;
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Could not play audio'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _seekTo(double value) async {
    final position = Duration(milliseconds: value.toInt());
    await _audioPlayer.seek(position);
  }

  Future<void> _stopAudio() async {
    await _audioPlayer.stop();
    setState(() {
      _isPlaying = false;
      _audioPosition = Duration.zero;
    });
  }

  Future<void> _changePlaybackSpeed(double speed) async {
    setState(() {
      _playbackSpeed = speed;
    });
    if (_isPlaying) {
      await _audioPlayer.setPlaybackRate(speed);
    }
  }

  Future<void> _saveToDownloads() async {
    if (_audioPath == null) return;

    try {
      // Request storage permission for older Android versions
      if (Platform.isAndroid) {
        var status = await Permission.storage.status;
        if (!status.isGranted) {
          status = await Permission.storage.request();
          if (!status.isGranted) {
            // Try without permission on Android 11+
          }
        }
      }
      
      // Get the Downloads directory
      final Directory downloadsDir = Directory('/storage/emulated/0/Download');
      
      if (await downloadsDir.exists()) {
        final String fileName = 'gujarati_tts_${DateTime.now().millisecondsSinceEpoch}.wav';
        final String newPath = '${downloadsDir.path}/$fileName';
        
        // Copy file to Downloads
        final File sourceFile = File(_audioPath!);
        await sourceFile.copy(newPath);
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Row(
                children: [
                  const Icon(Icons.check_circle, color: Colors.white),
                  const SizedBox(width: 8),
                  Expanded(child: Text('Saved: $fileName')),
                ],
              ),
              backgroundColor: Colors.green,
              duration: const Duration(seconds: 3),
            ),
          );
        }
      } else {
        throw Exception('Downloads folder not found');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Could not save. Try using Share instead.'),
            backgroundColor: Colors.orange,
          ),
        );
        // Fallback to share
        _shareAudio();
      }
    }
  }

  Future<void> _shareAudio() async {
    if (_audioPath == null) return;

    try {
      await Share.shareXFiles(
        [XFile(_audioPath!)],
        subject: 'Gujarati TTS Audio',
        text: 'Generated Gujarati speech audio',
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Could not share audio file'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));
    return '$minutes:$seconds';
  }

  String _formatTime(int seconds) {
    if (seconds < 60) {
      return '${seconds}s';
    } else {
      int mins = seconds ~/ 60;
      int secs = seconds % 60;
      return '${mins}m ${secs}s';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gujarati Vaani'),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.orange,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header Card
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.orange[50],
                        shape: BoxShape.circle,
                      ),
                      child: Icon(Icons.record_voice_over, size: 40, color: Colors.orange[700]),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Gujarati Text-to-Speech',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Type or paste Gujarati text',
                      style: TextStyle(color: Colors.grey[600]),
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 20),

            // Text Input
            TextField(
              controller: _textController,
              maxLines: 6,
              decoration: InputDecoration(
                hintText: 'ગુજરાતી ટેક્સ્ટ અહીં લખો...',
                hintStyle: TextStyle(color: Colors.grey[400]),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: BorderSide(color: Colors.grey[300]!),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: const BorderSide(color: Colors.orange, width: 2),
                ),
                filled: true,
                fillColor: Colors.grey[50],
                contentPadding: const EdgeInsets.all(16),
              ),
              style: const TextStyle(fontSize: 18, height: 1.5),
              onChanged: (value) {
                setState(() {});
              },
            ),

            // Character count and estimated time
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    '${_textController.text.length} characters',
                    style: TextStyle(color: Colors.grey[500], fontSize: 13),
                  ),
                  if (_textController.text.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.orange[50],
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        'Est: ~${_formatTime(_calculateEstimatedTime(_textController.text))}',
                        style: TextStyle(color: Colors.orange[700], fontSize: 13, fontWeight: FontWeight.w500),
                      ),
                    ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // Generate button with progress
            SizedBox(
              height: _isLoading ? 100 : 56,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _generateAudio,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange,
                  foregroundColor: Colors.white,
                  disabledBackgroundColor: Colors.orange,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  elevation: 2,
                ),
                child: _isLoading
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2.5,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(width: 12),
                              Text(
                                _statusMessage,
                                style: const TextStyle(fontSize: 16, color: Colors.white),
                              ),
                            ],
                          ),
                          const SizedBox(height: 10),
                          Text(
                            'Time: ${_formatTime(_elapsedSeconds)}',
                            style: const TextStyle(fontSize: 13, color: Colors.white70),
                          ),
                          const SizedBox(height: 6),
                          SizedBox(
                            width: 220,
                            height: 4,
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(2),
                              child: const LinearProgressIndicator(
                                backgroundColor: Colors.white24,
                                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                              ),
                            ),
                          ),
                        ],
                      )
                    : const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.graphic_eq, size: 24),
                          SizedBox(width: 10),
                          Text('Generate Speech', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                        ],
                      ),
              ),
            ),

            // Audio Player Card (shown after generation)
            if (_audioGenerated && _audioPath != null) ...[
              const SizedBox(height: 24),
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                color: Colors.orange[50],
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    children: [
                      // Player controls
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          // Stop button
                          IconButton(
                            onPressed: _stopAudio,
                            icon: const Icon(Icons.stop_rounded),
                            iconSize: 36,
                            color: Colors.orange[700],
                          ),
                          const SizedBox(width: 8),
                          // Play/Pause button
                          Container(
                            decoration: BoxDecoration(
                              color: Colors.orange,
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.orange.withOpacity(0.4),
                                  blurRadius: 12,
                                  offset: const Offset(0, 4),
                                ),
                              ],
                            ),
                            child: IconButton(
                              onPressed: _playAudio,
                              icon: Icon(_isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded),
                              iconSize: 48,
                              color: Colors.white,
                            ),
                          ),
                          const SizedBox(width: 8),
                          // Download button
                          IconButton(
                            onPressed: _saveToDownloads,
                            icon: const Icon(Icons.download_rounded),
                            iconSize: 36,
                            color: Colors.orange[700],
                            tooltip: 'Save to Downloads',
                          ),
                          const SizedBox(width: 8),
                          // Share button
                          IconButton(
                            onPressed: _shareAudio,
                            icon: const Icon(Icons.share_rounded),
                            iconSize: 36,
                            color: Colors.orange[700],
                            tooltip: 'Share',
                          ),
                        ],
                      ),
                      
                      const SizedBox(height: 16),
                      
                      // Seek bar
                      Column(
                        children: [
                          SliderTheme(
                            data: SliderTheme.of(context).copyWith(
                              activeTrackColor: Colors.orange,
                              inactiveTrackColor: Colors.orange[200],
                              thumbColor: Colors.orange[700],
                              overlayColor: Colors.orange.withOpacity(0.2),
                              trackHeight: 4,
                            ),
                            child: Slider(
                              value: _audioPosition.inMilliseconds.toDouble(),
                              max: _audioDuration.inMilliseconds.toDouble() > 0 
                                  ? _audioDuration.inMilliseconds.toDouble() 
                                  : 1.0,
                              onChanged: _seekTo,
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 16),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  _formatDuration(_audioPosition),
                                  style: TextStyle(color: Colors.grey[700], fontSize: 13),
                                ),
                                Text(
                                  _formatDuration(_audioDuration),
                                  style: TextStyle(color: Colors.grey[700], fontSize: 13),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      
                      const SizedBox(height: 16),
                      
                      // Speed control
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'Playback Speed',
                                  style: TextStyle(
                                    fontWeight: FontWeight.w600,
                                    color: Colors.grey[800],
                                  ),
                                ),
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.orange,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text(
                                    '${_playbackSpeed.toStringAsFixed(1)}x',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            SingleChildScrollView(
                              scrollDirection: Axis.horizontal,
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0].map((speed) {
                                  final isSelected = _playbackSpeed == speed;
                                  return Padding(
                                    padding: const EdgeInsets.symmetric(horizontal: 4),
                                    child: Material(
                                      color: isSelected ? Colors.orange : Colors.grey[100],
                                      borderRadius: BorderRadius.circular(8),
                                      child: InkWell(
                                        onTap: () => _changePlaybackSpeed(speed),
                                        borderRadius: BorderRadius.circular(8),
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                          decoration: BoxDecoration(
                                            borderRadius: BorderRadius.circular(8),
                                            border: Border.all(
                                              color: isSelected ? Colors.orange : Colors.grey[300]!,
                                            ),
                                          ),
                                          child: Text(
                                            '${speed}x',
                                            style: TextStyle(
                                              fontSize: 13,
                                              color: isSelected ? Colors.white : Colors.grey[700],
                                              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                                            ),
                                          ),
                                        ),
                                      ),
                                    ),
                                  );
                                }).toList(),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],

            const SizedBox(height: 20),

            // Info card
            Card(
              color: Colors.green[50],
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(14.0),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.green[100],
                        shape: BoxShape.circle,
                      ),
                      child: Icon(Icons.cloud_done, color: Colors.green[700], size: 20),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Cloud Powered',
                            style: TextStyle(
                              color: Colors.green[900],
                              fontWeight: FontWeight.w600,
                              fontSize: 14,
                            ),
                          ),
                          Text(
                            'Fine-tuned Gujarati TTS model',
                            style: TextStyle(color: Colors.green[700], fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
