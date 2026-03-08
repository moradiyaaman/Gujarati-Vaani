import 'dart:io';
import 'package:syncfusion_flutter_pdf/pdf.dart';

/// Service to extract text from PDF files (supports Gujarati)
class PdfService {
  // ── Common Gujarati chapter heading patterns ──
  static final _chapterPatterns = [
    // "પ્રકરણ ૧" / "પ્રકરણ 1" / "પ્રકરણ - ૧"
    RegExp(r'પ્રકરણ\s*[-–—:]?\s*[૦-૯0-9]+', caseSensitive: false),
    // "અધ્યાય ૧"
    RegExp(r'અધ્યાય\s*[-–—:]?\s*[૦-૯0-9]+', caseSensitive: false),
    // "ભાગ ૧"
    RegExp(r'ભાગ\s*[-–—:]?\s*[૦-૯0-9]+', caseSensitive: false),
    // English "Chapter 1" / "CHAPTER 1"
    RegExp(r'chapter\s*[-–—:]?\s*[0-9]+', caseSensitive: false),
    // Numbered headings like "1." "૧." at the start of a line with title text
    RegExp(r'^[૦-૯0-9]+\.\s+[\u0A80-\u0AFF]', multiLine: true),
  ];

  /// Extract all text from a PDF file, split into pages
  /// For large PDFs, limits to maxPages to prevent OOM
  static Future<List<String>> extractTextFromPdf(String filePath, {int? maxPages}) async {
    try {
      final file = File(filePath);
      final bytes = await file.readAsBytes();
      final PdfDocument document = PdfDocument(inputBytes: bytes);
      final PdfTextExtractor extractor = PdfTextExtractor(document);

      final totalPages = document.pages.count;
      final pagesToExtract = maxPages != null && maxPages < totalPages ? maxPages : totalPages;

      List<String> pages = [];
      for (int i = 0; i < pagesToExtract; i++) {
        final String text = extractor.extractText(startPageIndex: i, endPageIndex: i);
        final cleaned = _cleanExtractedText(text);
        if (cleaned.isNotEmpty) {
          pages.add(cleaned);
        }
      }

      document.dispose();
      return pages;
    } catch (e) {
      print('PDF extraction error: $e');
      return [];
    }
  }

  /// Get total page count without extracting text
  static Future<int> getPageCount(String filePath) async {
    try {
      final file = File(filePath);
      final bytes = await file.readAsBytes();
      final PdfDocument document = PdfDocument(inputBytes: bytes);
      final count = document.pages.count;
      document.dispose();
      return count;
    } catch (e) {
      return 0;
    }
  }

  /// Extract text for a specific range of pages (for on-demand loading)
  static Future<String> extractPageRange(String filePath, int startPage, int endPage) async {
    try {
      final file = File(filePath);
      final bytes = await file.readAsBytes();
      final PdfDocument document = PdfDocument(inputBytes: bytes);
      final PdfTextExtractor extractor = PdfTextExtractor(document);

      StringBuffer text = StringBuffer();
      final actualEnd = endPage < document.pages.count ? endPage : document.pages.count - 1;
      for (int i = startPage; i <= actualEnd; i++) {
        final pageText = _cleanExtractedText(
          extractor.extractText(startPageIndex: i, endPageIndex: i),
        );
        if (pageText.isNotEmpty) {
          if (text.isNotEmpty) text.write('\n\n');
          text.write(pageText);
        }
      }

      document.dispose();
      return text.toString();
    } catch (e) {
      print('PDF page range extraction error: $e');
      return '';
    }
  }

  // ─────────────────────────────────────────────
  //  TEXT CLEANING
  // ─────────────────────────────────────────────

  /// Clean raw extracted text: fix whitespace, remove junk, normalize punctuation
  static String _cleanExtractedText(String raw) {
    if (raw.trim().isEmpty) return '';

    String text = raw;

    // 1. Normalize line endings
    text = text.replaceAll('\r\n', '\n').replaceAll('\r', '\n');

    // 2. Remove null / control characters (except \n and \t)
    text = text.replaceAll(RegExp(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'), '');

    // 3. Remove common PDF artifacts: form-feed, soft-hyphen, ZWNJ/ZWJ noise
    text = text.replaceAll('\u000C', ''); // form-feed (page break)
    text = text.replaceAll('\u00AD', ''); // soft-hyphen

    // 4. Fix broken Gujarati words: remove random spaces inside Gujarati character sequences
    //    e.g. "ગુ જ રા ત" → "ગુજરાત"
    text = _fixBrokenGujaratiWords(text);

    // 5. Collapse multiple spaces into one (but preserve newlines)
    text = text.replaceAll(RegExp(r'[^\S\n]+'), ' ');

    // 6. Remove lines that are only page numbers (e.g. "12", "- 5 -", "Page 3")
    text = text.replaceAll(
      RegExp(r'^[\s]*[-–—]?\s*(?:page|પૃષ્ઠ)?\s*[૦-૯0-9]+\s*[-–—]?\s*$',
          multiLine: true, caseSensitive: false),
      '',
    );

    // 7. Remove repeated header/footer lines (lines appearing identically on many pages)
    text = _removeRepeatedLines(text);

    // 8. Collapse 3+ consecutive blank lines into 2
    text = text.replaceAll(RegExp(r'\n{3,}'), '\n\n');

    // 9. Trim each line
    text = text.split('\n').map((l) => l.trim()).join('\n');

    // 10. Final trim
    text = text.trim();

    return text;
  }

  /// Fix broken Gujarati words where spaces got inserted between characters.
  /// Joins single Gujarati characters / matras that are separated by a space.
  static String _fixBrokenGujaratiWords(String text) {
    // Pattern: Gujarati char + space + Gujarati char (single spacing that breaks words)
    // Only join when both sides are Gujarati and neither is a full word by itself
    // We look for a Gujarati letter/matra followed by space then another Gujarati letter/matra
    // But we only join if the "word" on either side is very short (1-2 chars)
    final lines = text.split('\n');
    final result = <String>[];

    for (final line in lines) {
      final words = line.split(' ');
      if (words.length <= 1) {
        result.add(line);
        continue;
      }

      final buffer = StringBuffer();
      String pending = words[0];

      for (int i = 1; i < words.length; i++) {
        final prev = pending;
        final curr = words[i];

        // Only merge if previous fragment is 1-2 Gujarati chars and current starts with Gujarati
        if (_isShortGujaratiFragment(prev) && _startsWithGujarati(curr)) {
          pending = prev + curr;
        } else if (_endsWithGujarati(prev) && _isShortGujaratiFragment(curr)) {
          pending = prev + curr;
        } else {
          if (buffer.isNotEmpty) buffer.write(' ');
          buffer.write(prev);
          pending = curr;
        }
      }
      if (buffer.isNotEmpty) buffer.write(' ');
      buffer.write(pending);
      result.add(buffer.toString());
    }
    return result.join('\n');
  }

  static bool _isShortGujaratiFragment(String s) {
    if (s.isEmpty || s.length > 3) return false;
    for (final r in s.runes) {
      if (r >= 0x0A80 && r <= 0x0AFF) return true;
    }
    return false;
  }

  static bool _startsWithGujarati(String s) {
    if (s.isEmpty) return false;
    final first = s.runes.first;
    return first >= 0x0A80 && first <= 0x0AFF;
  }

  static bool _endsWithGujarati(String s) {
    if (s.isEmpty) return false;
    final last = s.runes.last;
    return last >= 0x0A80 && last <= 0x0AFF;
  }

  /// Remove lines that appear 3+ times (likely headers/footers)
  static String _removeRepeatedLines(String text) {
    final lines = text.split('\n');
    final counts = <String, int>{};
    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isNotEmpty && trimmed.length < 80) {
        counts[trimmed] = (counts[trimmed] ?? 0) + 1;
      }
    }
    // Lines appearing 3+ times are likely headers/footers
    final repeated = counts.entries
        .where((e) => e.value >= 3)
        .map((e) => e.key)
        .toSet();

    if (repeated.isEmpty) return text;
    return lines.where((l) => !repeated.contains(l.trim())).join('\n');
  }

  // ─────────────────────────────────────────────
  //  SMART CHAPTER DETECTION
  // ─────────────────────────────────────────────

  /// Try to extract PDF bookmarks as chapter structure
  static List<_Bookmark> _extractBookmarks(PdfDocument document) {
    try {
      final bookmarks = document.bookmarks;
      if (bookmarks.count == 0) return [];

      final result = <_Bookmark>[];
      for (int i = 0; i < bookmarks.count; i++) {
        final bm = bookmarks[i];
        final title = bm.title.trim();
        if (title.isNotEmpty) {
          // Try to get destination page
          int pageIndex = -1;
          try {
            final dest = bm.destination;
            if (dest != null) {
              pageIndex = document.pages.indexOf(dest.page);
            }
          } catch (_) {}
          // Try namedDestination if direct destination not available
          if (pageIndex < 0) {
            try {
              final named = bm.namedDestination;
              if (named != null && named.destination != null) {
                pageIndex = document.pages.indexOf(named.destination!.page);
              }
            } catch (_) {}
          }
          result.add(_Bookmark(title: title, pageIndex: pageIndex >= 0 ? pageIndex : i));
        }
      }
      return result;
    } catch (e) {
      print('Bookmark extraction error: $e');
      return [];
    }
  }

  /// Detect chapter boundaries from extracted text using heading patterns
  static List<_ChapterBoundary> _detectChaptersFromText(List<String> pages) {
    final boundaries = <_ChapterBoundary>[];

    for (int pageIdx = 0; pageIdx < pages.length; pageIdx++) {
      final pageText = pages[pageIdx];
      final lines = pageText.split('\n');

      for (int lineIdx = 0; lineIdx < lines.length && lineIdx < 10; lineIdx++) {
        final line = lines[lineIdx].trim();
        if (line.isEmpty) continue;

        for (final pattern in _chapterPatterns) {
          if (pattern.hasMatch(line)) {
            // Use the matched line (plus next line if short) as chapter title
            String title = line;
            if (title.length < 40 && lineIdx + 1 < lines.length) {
              final nextLine = lines[lineIdx + 1].trim();
              if (nextLine.isNotEmpty && nextLine.length < 60) {
                title = '$title - $nextLine';
              }
            }
            boundaries.add(_ChapterBoundary(
              title: title,
              startPage: pageIdx,
            ));
            break; // Don't match multiple patterns on same page
          }
        }
        // Only check first few lines of each page for headings
        if (boundaries.isNotEmpty && boundaries.last.startPage == pageIdx) break;
      }
    }

    return boundaries;
  }

  /// Main chapter extraction with smart detection
  static Future<List<PdfChapter>> extractChapters(String filePath, String bookTitle) async {
    try {
      final file = File(filePath);
      final bytes = await file.readAsBytes();
      final PdfDocument document = PdfDocument(inputBytes: bytes);
      final PdfTextExtractor extractor = PdfTextExtractor(document);
      final totalPages = document.pages.count;

      if (totalPages == 0) {
        document.dispose();
        return [];
      }

      // ── Strategy 1: Try PDF bookmarks first ──
      final bookmarks = _extractBookmarks(document);

      // ── Extract all page texts (with cleaning) ──
      final pages = <String>[];
      for (int i = 0; i < totalPages; i++) {
        final raw = extractor.extractText(startPageIndex: i, endPageIndex: i);
        pages.add(_cleanExtractedText(raw));
      }
      document.dispose();

      // Remove completely empty pages from consideration
      if (pages.every((p) => p.isEmpty)) return [];

      // ── Strategy 2: Detect chapters from text heading patterns ──
      final textBoundaries = _detectChaptersFromText(pages);

      // ── Choose best strategy ──
      List<PdfChapter> chapters;

      if (bookmarks.length >= 2) {
        // Use bookmarks-based chapters
        print('Using PDF bookmarks for chapters (${bookmarks.length} found)');
        chapters = _buildChaptersFromBookmarks(bookmarks, pages);
      } else if (textBoundaries.length >= 2) {
        // Use text-detected chapters
        print('Using text-detected chapters (${textBoundaries.length} found)');
        chapters = _buildChaptersFromBoundaries(textBoundaries, pages);
      } else {
        // Fallback: smart grouping by content
        print('Using smart content grouping');
        chapters = _buildSmartGroupedChapters(pages);
      }

      // Filter out chapters with no meaningful content
      chapters = chapters.where((c) => c.content.trim().length > 20).toList();

      // If still empty, create single chapter with all text
      if (chapters.isEmpty) {
        final allText = pages.where((p) => p.isNotEmpty).join('\n\n');
        if (allText.trim().length > 20) {
          chapters = [PdfChapter(title: 'સંપૂર્ણ પુસ્તક', content: allText)];
        }
      }

      return chapters;
    } catch (e) {
      print('Chapter extraction error: $e');
      return [];
    }
  }

  /// Build chapters from PDF bookmarks
  static List<PdfChapter> _buildChaptersFromBookmarks(
      List<_Bookmark> bookmarks, List<String> pages) {
    final chapters = <PdfChapter>[];

    for (int i = 0; i < bookmarks.length; i++) {
      final startPage = bookmarks[i].pageIndex;
      final endPage = (i + 1 < bookmarks.length)
          ? bookmarks[i + 1].pageIndex - 1
          : pages.length - 1;

      if (startPage >= pages.length) continue;
      final actualEnd = endPage >= pages.length ? pages.length - 1 : endPage;

      final content = StringBuffer();
      for (int p = startPage; p <= actualEnd; p++) {
        if (pages[p].isNotEmpty) {
          if (content.isNotEmpty) content.write('\n\n');
          content.write(pages[p]);
        }
      }

      if (content.isNotEmpty) {
        final pageRange = startPage == actualEnd
            ? '(પૃષ્ઠ ${_toGujaratiNum(startPage + 1)})'
            : '(પૃષ્ઠ ${_toGujaratiNum(startPage + 1)}-${_toGujaratiNum(actualEnd + 1)})';
        chapters.add(PdfChapter(
          title: '${bookmarks[i].title} $pageRange',
          content: content.toString(),
        ));
      }
    }

    return chapters;
  }

  /// Build chapters from text-detected boundaries
  static List<PdfChapter> _buildChaptersFromBoundaries(
      List<_ChapterBoundary> boundaries, List<String> pages) {
    final chapters = <PdfChapter>[];

    // Add any content before the first detected chapter as "પ્રસ્તાવના" (preface)
    if (boundaries.first.startPage > 0) {
      final prefaceContent = StringBuffer();
      for (int p = 0; p < boundaries.first.startPage; p++) {
        if (pages[p].isNotEmpty) {
          if (prefaceContent.isNotEmpty) prefaceContent.write('\n\n');
          prefaceContent.write(pages[p]);
        }
      }
      if (prefaceContent.length > 50) {
        chapters.add(PdfChapter(
          title: 'પ્રસ્તાવના',
          content: prefaceContent.toString(),
        ));
      }
    }

    for (int i = 0; i < boundaries.length; i++) {
      final startPage = boundaries[i].startPage;
      final endPage = (i + 1 < boundaries.length)
          ? boundaries[i + 1].startPage - 1
          : pages.length - 1;

      final actualEnd = endPage >= pages.length ? pages.length - 1 : endPage;

      final content = StringBuffer();
      for (int p = startPage; p <= actualEnd; p++) {
        if (pages[p].isNotEmpty) {
          if (content.isNotEmpty) content.write('\n\n');
          content.write(pages[p]);
        }
      }

      if (content.isNotEmpty) {
        chapters.add(PdfChapter(
          title: boundaries[i].title,
          content: content.toString(),
        ));
      }
    }

    return chapters;
  }

  /// Smart grouping: merge small pages together, split large content blocks
  /// Creates sensible chapters based on content size for good TTS segments
  static List<PdfChapter> _buildSmartGroupedChapters(List<String> pages) {
    final chapters = <PdfChapter>[];

    // Target: each chapter ~1000-3000 characters (good for TTS audio segments)
    const minChapterLen = 500;
    const targetChapterLen = 2000;
    const maxChapterLen = 4000;

    StringBuffer currentContent = StringBuffer();
    int startPage = -1;
    int currentPage = -1;

    for (int i = 0; i < pages.length; i++) {
      if (pages[i].isEmpty) continue;

      if (startPage < 0) startPage = i;
      currentPage = i;

      if (currentContent.isNotEmpty) currentContent.write('\n\n');
      currentContent.write(pages[i]);

      // Check if we should create a chapter break
      bool shouldBreak = false;

      if (currentContent.length >= targetChapterLen) {
        shouldBreak = true;
      }
      // Also break if next page would push us way over
      if (i + 1 < pages.length && 
          currentContent.length + pages[i + 1].length > maxChapterLen &&
          currentContent.length >= minChapterLen) {
        shouldBreak = true;
      }
      // Break at last page
      if (i == pages.length - 1 && currentContent.isNotEmpty) {
        shouldBreak = true;
      }

      if (shouldBreak && currentContent.isNotEmpty) {
        final chapterNum = chapters.length + 1;
        final pageRange = startPage == currentPage
            ? 'પૃષ્ઠ ${_toGujaratiNum(startPage + 1)}'
            : 'પૃષ્ઠ ${_toGujaratiNum(startPage + 1)}-${_toGujaratiNum(currentPage + 1)}';
        chapters.add(PdfChapter(
          title: 'પ્રકરણ ${_toGujaratiNum(chapterNum)} ($pageRange)',
          content: currentContent.toString(),
        ));
        currentContent = StringBuffer();
        startPage = -1;
      }
    }

    // Any remaining content
    if (currentContent.isNotEmpty) {
      final chapterNum = chapters.length + 1;
      chapters.add(PdfChapter(
        title: 'પ્રકરણ ${_toGujaratiNum(chapterNum)} (પૃષ્ઠ ${_toGujaratiNum(startPage + 1)}-${_toGujaratiNum(currentPage + 1)})',
        content: currentContent.toString(),
      ));
    }

    return chapters;
  }

  /// Convert number to Gujarati numerals
  static String _toGujaratiNum(int num) {
    const gujaratiDigits = ['૦', '૧', '૨', '૩', '૪', '૫', '૬', '૭', '૮', '૯'];
    return num.toString().split('').map((d) => gujaratiDigits[int.parse(d)]).join();
  }

  /// Check if text contains Gujarati characters (U+0A80 to U+0AFF)
  static bool isGujaratiText(String text) {
    if (text.isEmpty) return false;
    int gujaratiChars = 0;
    int totalChars = 0;
    for (final codeUnit in text.runes) {
      if (codeUnit == 32 || codeUnit == 10 || codeUnit == 13 || codeUnit == 9) continue;
      totalChars++;
      if (codeUnit >= 0x0A80 && codeUnit <= 0x0AFF) {
        gujaratiChars++;
      }
    }
    if (totalChars == 0) return false;
    final ratio = gujaratiChars / totalChars;
    print('Gujarati detection: $gujaratiChars/$totalChars chars = ${(ratio * 100).toStringAsFixed(1)}%');
    return ratio >= 0.05;
  }

  /// Quick check on first few pages only (for large PDFs)
  static Future<bool> quickGujaratiCheck(String filePath, {int maxPages = 5}) async {
    try {
      final file = File(filePath);
      final bytes = await file.readAsBytes();
      final PdfDocument document = PdfDocument(inputBytes: bytes);
      final extractor = PdfTextExtractor(document);

      StringBuffer sampleText = StringBuffer();
      final pagesToCheck = document.pages.count < maxPages ? document.pages.count : maxPages;
      for (int i = 0; i < pagesToCheck; i++) {
        final text = extractor.extractText(startPageIndex: i, endPageIndex: i);
        sampleText.write(text);
        if (sampleText.length > 500) break;
      }

      document.dispose();
      return isGujaratiText(sampleText.toString());
    } catch (e) {
      print('Quick Gujarati check error: $e');
      return true;
    }
  }

  /// Get file name without extension
  static String getBookName(String filePath) {
    final fileName = filePath.split('/').last.split('\\').last;
    return fileName.replaceAll('.pdf', '').replaceAll('.PDF', '').replaceAll('_', ' ');
  }
}

// ── Internal helper classes ──

class _Bookmark {
  final String title;
  final int pageIndex;
  _Bookmark({required this.title, required this.pageIndex});
}

class _ChapterBoundary {
  final String title;
  final int startPage;
  _ChapterBoundary({required this.title, required this.startPage});
}

/// Represents a chapter extracted from PDF
class PdfChapter {
  final String title;
  final String content;

  PdfChapter({required this.title, required this.content});
}
