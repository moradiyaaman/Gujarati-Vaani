import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'dart:async';
import 'package:path_provider/path_provider.dart';

/// Shared TTS service used by all screens
class TTSService {
  static const String apiUrl =
      'https://moradiyaaman-gujarati-vaani-tts.hf.space/synthesize';

  /// Check internet connection
  static Future<bool> checkInternet() async {
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

  /// Validate if text is Gujarati
  static bool isGujaratiText(String text) {
    final gujaratiPattern = RegExp(r'[\u0A80-\u0AFF]');
    return gujaratiPattern.hasMatch(text);
  }

  /// Detect predominant language
  static String detectLanguage(String text) {
    final gujaratiChars = RegExp(r'[\u0A80-\u0AFF]').allMatches(text).length;
    final hindiChars = RegExp(r'[\u0900-\u097F]').allMatches(text).length;
    final englishChars = RegExp(r'[a-zA-Z]').allMatches(text).length;
    final otherIndianChars = RegExp(
            r'[\u0980-\u09FF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]')
        .allMatches(text)
        .length;

    final cleanText = text.replaceAll(RegExp(r'[\s\p{P}\d]'), '');
    final totalChars = cleanText.length;

    if (totalChars == 0) return 'unknown';
    if (gujaratiChars > 0 && (gujaratiChars / totalChars) > 0.8) {
      return 'gujarati';
    }
    if (hindiChars > gujaratiChars && hindiChars > englishChars) return 'hindi';
    if (englishChars > gujaratiChars && englishChars > hindiChars) {
      return 'english';
    }
    if (otherIndianChars > gujaratiChars) return 'other_indian';
    if (gujaratiChars > 0) return 'mixed';
    return 'unknown';
  }

  /// Preprocess text for better TTS output
  static String preprocessText(String text) {
    String processed = text.trim();
    // Normalize whitespace
    processed = processed.replaceAll(RegExp(r'\s+'), ' ');
    // Replace ". " (English period) with "। " for consistency
    processed = processed.replaceAll('. ', '। ');
    // Ensure proper ending
    if (!processed.endsWith('।') &&
        !processed.endsWith('.') &&
        !processed.endsWith('!') &&
        !processed.endsWith('?')) {
      processed += '।';
    }
    return processed;
  }

  /// Split text into smaller chunks at sentence boundaries for better TTS
  static List<String> splitTextIntoChunks(String text, int maxChunkSize) {
    List<String> chunks = [];
    RegExp decimalPattern = RegExp(r'(\d+\.\d+|[૦-૯]+\.[૦-૯]+)');
    Map<String, String> decimalMap = {};
    int decimalIndex = 0;

    String protectedText = text.replaceAllMapped(decimalPattern, (match) {
      String placeholder = '<<<DECIMAL$decimalIndex>>>';
      decimalMap[placeholder] = match.group(0)!;
      decimalIndex++;
      return placeholder;
    });

    RegExp sentenceEnd = RegExp(r'।|\.(?=\s|$)|!(?=\s|$)|\?(?=\s|$)');
    List<String> parts = protectedText.split(sentenceEnd);
    List<Match> delimiters = sentenceEnd.allMatches(protectedText).toList();

    String currentChunk = '';
    for (int i = 0; i < parts.length; i++) {
      String sentence = parts[i].trim();
      if (sentence.isEmpty) continue;
      decimalMap.forEach((placeholder, original) {
        sentence = sentence.replaceAll(placeholder, original);
      });
      String delimiter =
          (i < delimiters.length) ? delimiters[i].group(0)! : '';
      String sentenceWithPunct =
          sentence + (delimiter.isNotEmpty ? delimiter : '।') + ' ';
      if ((currentChunk + sentenceWithPunct).length > maxChunkSize &&
          currentChunk.isNotEmpty) {
        chunks.add(currentChunk.trim());
        currentChunk = sentenceWithPunct;
      } else {
        currentChunk += sentenceWithPunct;
      }
    }
    if (currentChunk.trim().isNotEmpty) chunks.add(currentChunk.trim());
    if (chunks.isEmpty && text.isNotEmpty) {
      // Try splitting on commas
      final commaParts = text.split(RegExp(r'[,،]'));
      String buffer = '';
      for (var part in commaParts) {
        if ((buffer + part).length > maxChunkSize && buffer.isNotEmpty) {
          chunks.add(buffer.trim());
          buffer = part;
        } else {
          buffer += (buffer.isEmpty ? '' : ', ') + part;
        }
      }
      if (buffer.trim().isNotEmpty) chunks.add(buffer.trim());
      // Last resort: hard split
      if (chunks.isEmpty) {
        for (int i = 0; i < text.length; i += maxChunkSize) {
          int end =
              (i + maxChunkSize < text.length) ? i + maxChunkSize : text.length;
          chunks.add(text.substring(i, end));
        }
      }
    }
    return chunks;
  }

  /// Synthesize text to audio file, returns file path.
  /// Uses speed 0.9 by default for clearer Gujarati speech.
  static Future<String?> synthesize(
    String text, {
    double speed = 0.9,
    void Function(String status)? onStatus,
    void Function(int elapsed)? onElapsed,
  }) async {
    // Preprocess text for better TTS quality
    String processedText = preprocessText(text);

    // Use smaller chunks (250 chars) for better pronunciation
    const int maxChunkSize = 250;
    List<String> chunks;

    if (processedText.length > maxChunkSize) {
      chunks = splitTextIntoChunks(processedText, maxChunkSize);
    } else {
      chunks = [processedText];
    }

    List<List<int>> audioBytes = [];
    int retryCount = 0;
    const int maxRetries = 2;

    for (int i = 0; i < chunks.length; i++) {
      onStatus?.call('Generating ${i + 1}/${chunks.length}...');

      try {
        final response = await http
            .post(
              Uri.parse(apiUrl),
              headers: {'Content-Type': 'application/json; charset=utf-8'},
              body: json.encode({'text': chunks[i], 'speed': speed}),
            )
            .timeout(const Duration(seconds: 120));

        if (response.statusCode == 200) {
          audioBytes.add(response.bodyBytes);
          retryCount = 0;
        } else if (response.statusCode == 503) {
          if (retryCount < maxRetries) {
            onStatus?.call('Server waking up, please wait...');
            await Future.delayed(const Duration(seconds: 15));
            retryCount++;
            i--;
            continue;
          } else {
            return null;
          }
        } else {
          throw Exception('Server error: ${response.statusCode}');
        }
      } on TimeoutException {
        if (retryCount < maxRetries) {
          onStatus?.call('Retrying ${i + 1}/${chunks.length}...');
          retryCount++;
          i--;
          await Future.delayed(const Duration(seconds: 3));
          continue;
        } else {
          return null;
        }
      } catch (e) {
        if (retryCount < maxRetries) {
          retryCount++;
          i--;
          await Future.delayed(const Duration(seconds: 3));
          continue;
        }
        return null;
      }
    }

    if (audioBytes.isEmpty) return null;

    onStatus?.call('Preparing audio...');

    try {
      final directory = await getTemporaryDirectory();
      final filePath =
          '${directory.path}/gujarati_tts_${DateTime.now().millisecondsSinceEpoch}.wav';

      List<int> combinedAudio = [];
      if (audioBytes.length == 1) {
        combinedAudio = audioBytes[0];
      } else {
        // Combine WAV files: keep first header, append data from rest
        combinedAudio.addAll(audioBytes[0]);

        // Add short silence between chunks for natural pauses
        // 16000 Hz * 2 bytes * 0.15s = 4800 bytes of silence
        final silencePadding = List<int>.filled(4800, 0);

        for (int i = 1; i < audioBytes.length; i++) {
          combinedAudio.addAll(silencePadding);
          if (audioBytes[i].length > 44) {
            combinedAudio.addAll(audioBytes[i].sublist(44));
          }
        }

        // Update WAV header sizes
        int dataSize = combinedAudio.length - 44;
        int fileSize = combinedAudio.length - 8;
        combinedAudio[4] = fileSize & 0xFF;
        combinedAudio[5] = (fileSize >> 8) & 0xFF;
        combinedAudio[6] = (fileSize >> 16) & 0xFF;
        combinedAudio[7] = (fileSize >> 24) & 0xFF;
        combinedAudio[40] = dataSize & 0xFF;
        combinedAudio[41] = (dataSize >> 8) & 0xFF;
        combinedAudio[42] = (dataSize >> 16) & 0xFF;
        combinedAudio[43] = (dataSize >> 24) & 0xFF;
      }

      final file = File(filePath);
      await file.writeAsBytes(combinedAudio);
      return filePath;
    } catch (e) {
      return null;
    }
  }
}
