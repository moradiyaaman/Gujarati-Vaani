import 'package:flutter/material.dart';
import 'dart:io';
import 'dart:async';
import 'package:audioplayers/audioplayers.dart';
import 'package:share_plus/share_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/tts_service.dart';

/// Sample Gujarati books with chapters
class GujaratiBook {
  final String title;
  final String author;
  final String coverEmoji;
  final List<BookChapter> chapters;

  const GujaratiBook({
    required this.title,
    required this.author,
    required this.coverEmoji,
    required this.chapters,
  });
}

class BookChapter {
  final String title;
  final String content;

  const BookChapter({required this.title, required this.content});
}

// Sample Gujarati books - shorter sentences for better TTS
final List<GujaratiBook> sampleBooks = [
  GujaratiBook(
    title: 'ગુજરાતી વાર્તાઓ',
    author: 'લોક સાહિત્ય',
    coverEmoji: '📖',
    chapters: [
      BookChapter(
        title: 'પ્રકરણ ૧: ખેડૂત અને સાપ',
        content:
            'એક ગામમાં એક ખેડૂત રહેતો હતો। તે ખૂબ મહેનતુ અને દયાળુ હતો। '
            'એક ઠંડીની રાત્રે તે ખેતરમાંથી ઘરે પાછો ફરી રહ્યો હતો। '
            'રસ્તામાં તેને એક સાપ દેખાયો। સાપ ઠંડીથી ઠૂંઠવાઈ ગયો હતો। '
            'ખેડૂતને તેના પર દયા આવી। તેણે સાપને ઉઠાવીને પોતાના ખોળામાં મૂક્યો। '
            'સાપ ગરમ થયો એટલે તેણે ખેડૂતને ડંખ માર્યો। '
            'ખેડૂતે કહ્યું કે મેં તારી મદદ કરી અને તેં મને ડંખ માર્યો। '
            'સાપે કહ્યું કે આ મારો સ્વભાવ છે। '
            'આ વાર્તાથી આપણે શીખીએ છીએ કે દુષ્ટ લોકો ક્યારેય પોતાનો સ્વભાવ બદલતા નથી।',
      ),
      BookChapter(
        title: 'પ્રકરણ ૨: ચતુર શિયાળ',
        content:
            'એક જંગલમાં એક ચતુર શિયાળ રહેતું હતું। '
            'એક દિવસ તેને ભૂખ લાગી। તે ખોરાકની શોધમાં નીકળ્યું। '
            'રસ્તામાં તેને એક ઝાડ પર દ્રાક્ષ દેખાઈ। દ્રાક્ષ ખૂબ ઊંચી હતી। '
            'શિયાળે ઘણી વાર કૂદકો માર્યો। પણ દ્રાક્ષ પકડી ન શક્યું। '
            'છેવટે થાકીને શિયાળે કહ્યું કે આ દ્રાક્ષ ખાટી છે। '
            'મારે નથી ખાવી। '
            'આ વાર્તાથી આપણે શીખીએ છીએ। '
            'જ્યારે આપણે કોઈ વસ્તુ મેળવી ન શકીએ ત્યારે આપણે તેને ખરાબ કહીએ છીએ।',
      ),
      BookChapter(
        title: 'પ્રકરણ ૩: કાગડો અને લોમડી',
        content:
            'એક કાગડો ઝાડ પર બેઠો હતો। તેના મોઢામાં રોટલીનો ટુકડો હતો। '
            'ત્યાં એક લોમડી આવી અને કાગડાને જોયો। '
            'લોમડીને ભૂખ લાગી હતી। '
            'તેણે કાગડાને કહ્યું કે તમે ખૂબ સુંદર છો। '
            'તમારો અવાજ પણ ખૂબ મધુર છે। કૃપા કરીને એક ગીત ગાઓ। '
            'કાગડો ખુશ થયો અને ગાવા લાગ્યો। '
            'જેવો તેણે મોઢું ખોલ્યું કે રોટલીનો ટુકડો નીચે પડ્યો। '
            'લોમડીએ ઝડપથી રોટલી ઉપાડી અને ભાગી ગઈ। '
            'આ વાર્તાથી આપણે શીખીએ છીએ કે ખુશામત કરનારાઓથી સાવધાન રહેવું જોઈએ।',
      ),
    ],
  ),
  GujaratiBook(
    title: 'ગુજરાતી કવિતાઓ',
    author: 'પ્રસિદ્ધ કવિઓ',
    coverEmoji: '📝',
    chapters: [
      BookChapter(
        title: 'વૈષ્ણવ જન તો',
        content:
            'વૈષ્ણવ જન તો તેને કહીએ જે પીડ પરાઈ જાણે રે। '
            'પરદુઃખે ઉપકાર કરે તોયે મન અભિમાન ન આણે રે। '
            'સકળ લોકમાં સહુને વંદે। નિંદા ન કરે કેની રે। '
            'વાચ કાછ મન નિશ્ચળ રાખે। ધન ધન જનની તેની રે। '
            'સમદૃષ્ટિ ને તૃષ્ણા ત્યાગી। પરસ્ત્રી જેને માત રે। '
            'જિહ્વા થકી અસત્ય ન બોલે। પરધન નવ ઝાલે હાથ રે।',
      ),
      BookChapter(
        title: 'છેલ્લો કટોરો',
        content:
            'અમે મૂઆ ત્યાં ઝાડવાં ઊભાં રહેશે। '
            'અમે મૂઆ ત્યાં પંખીડાં ગાશે। '
            'અમે મૂઆ ત્યાં ઝરણાંઓ વહેશે। '
            'અમે મૂઆ ત્યાં ફૂલો ખીલશે। '
            'અમે મૂઆ ત્યાં પવન ફૂંકાશે। '
            'અમે મૂઆ ત્યાં તારા ચમકશે। '
            'પ્રકૃતિ અમર છે અને અમે ફાની છીએ। '
            'પણ અમારા શબ્દો અમર રહેશે।',
      ),
    ],
  ),
  GujaratiBook(
    title: 'ગાંધીજીની આત્મકથા',
    author: 'મોહનદાસ ક. ગાંધી',
    coverEmoji: '🕊️',
    chapters: [
      BookChapter(
        title: 'પ્રકરણ ૧: જન્મ અને બાળપણ',
        content:
            'ગાંધી પરિવાર કાઠિયાવાડના રાજકોટ અને પોરબંદર રાજ્યમાં રહેતો હતો। '
            'છેલ્લી ત્રણ પેઢીથી ગાંધી પરિવારના સભ્યો દીવાન હતા। '
            'મારા દાદા ઉત્તમચંદ ગાંધી રાજ્યના દીવાન હતા। '
            'મારા પિતા કરમચંદ ગાંધી પણ રાજકોટના દીવાન હતા। '
            'મારી માતા પુતળીબાઈ ખૂબ ધાર્મિક હતાં। '
            'તેઓ દરરોજ મંદિરે જતાં। વ્રત કરવા એ તેમના માટે સામાન્ય વાત હતી। '
            'મારો જન્મ બીજી ઓક્ટોબર ઓગણીસસો ઓગણસત્તરમાં પોરબંદરમાં થયો હતો।',
      ),
      BookChapter(
        title: 'પ્રકરણ ૨: શાળાના દિવસો',
        content:
            'મારું શાળાનું જીવન સુખી નહોતું। '
            'હું ખૂબ શરમાળ છોકરો હતો। '
            'ઘંટડી વાગે ત્યારે દોડીને શાળાએ જતો। '
            'શાળા છૂટે એટલે દોડીને ઘરે આવતો। '
            'કોઈની સાથે વાત કરતો ન હતો। '
            'મને ડર લાગતો કે કોઈ મારી મજાક ઉડાવશે। '
            'પુસ્તકો અને પાઠ એ જ મારા સાથી હતા। '
            'એક વાર ઇન્સ્પેક્ટર અમારી શાળામાં આવ્યા હતા। '
            'તેમણે અમને અંગ્રેજીના પાંચ શબ્દો લખવા કહ્યું। '
            'એક શબ્દ મેં ખોટો લખ્યો। '
            'શિક્ષકે મને ઈશારો કર્યો કે બાજુવાળાનું જોઈને લખ। '
            'પણ મેં ન જોયું। મને લાગ્યું કે નકલ કરવી ખોટી છે।',
      ),
    ],
  ),
];

class BookReaderScreen extends StatefulWidget {
  const BookReaderScreen({super.key});

  @override
  State<BookReaderScreen> createState() => _BookReaderScreenState();
}

class _BookReaderScreenState extends State<BookReaderScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: sampleBooks.length + 1,
        itemBuilder: (context, index) {
          if (index == 0) {
            return _buildHeader();
          }
          final book = sampleBooks[index - 1];
          return _buildBookCard(book);
        },
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.orange[50],
              shape: BoxShape.circle,
            ),
            child:
                Icon(Icons.auto_stories, size: 40, color: Colors.orange[700]),
          ),
          const SizedBox(height: 12),
          const Text(
            'ગુજરાતી પુસ્તકો',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Text(
            'Select a book to listen chapter-wise',
            style: TextStyle(color: Colors.grey[600], fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _buildBookCard(GujaratiBook book) {
    return Card(
      elevation: 3,
      margin: const EdgeInsets.only(bottom: 16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ChapterListScreen(book: book),
            ),
          );
        },
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                width: 70,
                height: 90,
                decoration: BoxDecoration(
                  color: Colors.orange[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.orange[200]!),
                ),
                child: Center(
                  child: Text(book.coverEmoji,
                      style: const TextStyle(fontSize: 36)),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      book.title,
                      style: const TextStyle(
                          fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      book.author,
                      style: TextStyle(color: Colors.grey[600], fontSize: 14),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.orange[50],
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        '${book.chapters.length} chapters',
                        style: TextStyle(
                            color: Colors.orange[700],
                            fontSize: 12,
                            fontWeight: FontWeight.w500),
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: Colors.orange[400], size: 28),
            ],
          ),
        ),
      ),
    );
  }
}

/// Chapter list screen
class ChapterListScreen extends StatelessWidget {
  final GujaratiBook book;
  const ChapterListScreen({super.key, required this.book});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(book.title),
        backgroundColor: Colors.orange,
        foregroundColor: Colors.white,
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: book.chapters.length,
        itemBuilder: (context, index) {
          final chapter = book.chapters[index];
          return Card(
            elevation: 2,
            margin: const EdgeInsets.only(bottom: 12),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            child: InkWell(
              borderRadius: BorderRadius.circular(14),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => ChapterReaderScreen(
                      book: book,
                      chapterIndex: index,
                    ),
                  ),
                );
              },
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: Colors.orange[100],
                        shape: BoxShape.circle,
                      ),
                      child: Center(
                        child: Text(
                          '${index + 1}',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.orange[800],
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            chapter.title,
                            style: const TextStyle(
                                fontSize: 16, fontWeight: FontWeight.w600),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '${chapter.content.length} characters',
                            style: TextStyle(
                                color: Colors.grey[500], fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                    Icon(Icons.headphones, color: Colors.orange[400]),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

/// Chapter reader with TTS playback
class ChapterReaderScreen extends StatefulWidget {
  final GujaratiBook book;
  final int chapterIndex;

  const ChapterReaderScreen({
    super.key,
    required this.book,
    required this.chapterIndex,
  });

  @override
  State<ChapterReaderScreen> createState() => _ChapterReaderScreenState();
}

class _ChapterReaderScreenState extends State<ChapterReaderScreen> {
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool _isLoading = false;
  bool _isPlaying = false;
  String? _audioPath;
  String _statusMessage = '';
  Duration _audioDuration = Duration.zero;
  Duration _audioPosition = Duration.zero;
  double _playbackSpeed = 1.0;
  Timer? _progressTimer;
  int _elapsedSeconds = 0;

  BookChapter get chapter => widget.book.chapters[widget.chapterIndex];

  @override
  void initState() {
    super.initState();
    _setupAudioPlayer();
  }

  void _setupAudioPlayer() {
    _audioPlayer.onDurationChanged.listen((d) {
      if (mounted) setState(() => _audioDuration = d);
    });
    _audioPlayer.onPositionChanged.listen((p) {
      if (mounted) setState(() => _audioPosition = p);
    });
    _audioPlayer.onPlayerComplete.listen((_) {
      if (mounted) {
        setState(() {
          _isPlaying = false;
          _audioPosition = Duration.zero;
        });
        // Auto-play next chapter if available
        if (widget.chapterIndex < widget.book.chapters.length - 1) {
          _showNextChapterDialog();
        }
      }
    });
  }

  void _showNextChapterDialog() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Chapter Complete!'),
        content: const Text('Play next chapter?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('No'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.of(ctx).pop();
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                  builder: (context) => ChapterReaderScreen(
                    book: widget.book,
                    chapterIndex: widget.chapterIndex + 1,
                  ),
                ),
              );
            },
            style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
            child: const Text('Next Chapter',
                style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    _progressTimer?.cancel();
    super.dispose();
  }

  Future<void> _generateChapterAudio() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Connecting...';
      _elapsedSeconds = 0;
    });

    bool hasInternet = await TTSService.checkInternet();
    if (!hasInternet) {
      setState(() {
        _isLoading = false;
        _statusMessage = '';
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('No internet connection'),
            backgroundColor: Colors.red,
          ),
        );
      }
      return;
    }

    _progressTimer?.cancel();
    _progressTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _elapsedSeconds++);
    });

    final path = await TTSService.synthesize(
      chapter.content,
      onStatus: (s) {
        if (mounted) setState(() => _statusMessage = s);
      },
    );

    _progressTimer?.cancel();

    if (path != null && mounted) {
      setState(() {
        _audioPath = path;
        _isLoading = false;
        _statusMessage = '';
      });
      await _playAudio();
    } else if (mounted) {
      setState(() {
        _isLoading = false;
        _statusMessage = '';
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Failed to generate audio. Try again.'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _playAudio() async {
    if (_audioPath == null) return;
    try {
      if (_isPlaying) {
        await _audioPlayer.pause();
        setState(() => _isPlaying = false);
      } else {
        await _audioPlayer.setPlaybackRate(_playbackSpeed);
        await _audioPlayer.play(DeviceFileSource(_audioPath!));
        setState(() => _isPlaying = true);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Could not play audio'),
              backgroundColor: Colors.red),
        );
      }
    }
  }

  Future<void> _saveToDownloads() async {
    if (_audioPath == null) return;
    try {
      if (Platform.isAndroid) {
        var status = await Permission.storage.status;
        if (!status.isGranted) await Permission.storage.request();
      }
      final downloadsDir = Directory('/storage/emulated/0/Download');
      if (await downloadsDir.exists()) {
        final fileName =
            'chapter_${widget.chapterIndex + 1}_${DateTime.now().millisecondsSinceEpoch}.wav';
        final newPath = '${downloadsDir.path}/$fileName';
        await File(_audioPath!).copy(newPath);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Saved: $fileName'),
              backgroundColor: Colors.green,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Could not save file'),
              backgroundColor: Colors.red),
        );
      }
    }
  }

  String _formatDuration(Duration d) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    return '${twoDigits(d.inMinutes.remainder(60))}:${twoDigits(d.inSeconds.remainder(60))}';
  }

  String _formatTime(int seconds) {
    if (seconds < 60) return '${seconds}s';
    return '${seconds ~/ 60}m ${seconds % 60}s';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Chapter ${widget.chapterIndex + 1}'),
        backgroundColor: Colors.orange,
        foregroundColor: Colors.white,
        actions: [
          if (_audioPath != null)
            IconButton(
              icon: const Icon(Icons.download),
              onPressed: _saveToDownloads,
              tooltip: 'Save audio',
            ),
          if (_audioPath != null)
            IconButton(
              icon: const Icon(Icons.share),
              onPressed: () async {
                if (_audioPath != null) {
                  await Share.shareXFiles([XFile(_audioPath!)],
                      subject: 'Gujarati Book Audio');
                }
              },
              tooltip: 'Share audio',
            ),
        ],
      ),
      body: Column(
        children: [
          // Chapter text (scrollable)
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    chapter.title,
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    chapter.content,
                    style: const TextStyle(
                      fontSize: 18,
                      height: 1.8,
                      color: Colors.black87,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Audio controls at bottom
          Container(
            decoration: BoxDecoration(
              color: Colors.orange[50],
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 10,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (_isLoading) ...[
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                            strokeWidth: 2.5, color: Colors.orange),
                      ),
                      const SizedBox(width: 12),
                      Text(_statusMessage,
                          style: const TextStyle(fontSize: 14)),
                      const SizedBox(width: 8),
                      Text('(${_formatTime(_elapsedSeconds)})',
                          style: TextStyle(
                              fontSize: 12, color: Colors.grey[600])),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const LinearProgressIndicator(color: Colors.orange),
                ] else if (_audioPath != null) ...[
                  // Seek bar
                  SliderTheme(
                    data: SliderTheme.of(context).copyWith(
                      activeTrackColor: Colors.orange,
                      inactiveTrackColor: Colors.orange[200],
                      thumbColor: Colors.orange[700],
                      trackHeight: 4,
                    ),
                    child: Slider(
                      value: _audioPosition.inMilliseconds.toDouble(),
                      max: _audioDuration.inMilliseconds.toDouble() > 0
                          ? _audioDuration.inMilliseconds.toDouble()
                          : 1.0,
                      onChanged: (v) =>
                          _audioPlayer.seek(Duration(milliseconds: v.toInt())),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(_formatDuration(_audioPosition),
                            style: const TextStyle(fontSize: 12)),
                        Text(_formatDuration(_audioDuration),
                            style: const TextStyle(fontSize: 12)),
                      ],
                    ),
                  ),
                  const SizedBox(height: 8),
                  // Controls
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Speed selector
                      PopupMenuButton<double>(
                        onSelected: (speed) {
                          setState(() => _playbackSpeed = speed);
                          if (_isPlaying) _audioPlayer.setPlaybackRate(speed);
                        },
                        itemBuilder: (_) => [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                            .map((s) => PopupMenuItem(
                                value: s, child: Text('${s}x')))
                            .toList(),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: Colors.orange[100],
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text('${_playbackSpeed}x',
                              style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  color: Colors.orange[800])),
                        ),
                      ),
                      const SizedBox(width: 16),
                      // Stop
                      IconButton(
                        onPressed: () async {
                          await _audioPlayer.stop();
                          setState(() {
                            _isPlaying = false;
                            _audioPosition = Duration.zero;
                          });
                        },
                        icon: const Icon(Icons.stop_rounded),
                        iconSize: 36,
                        color: Colors.orange[700],
                      ),
                      // Play/Pause
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.orange,
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                                color: Colors.orange.withOpacity(0.4),
                                blurRadius: 8)
                          ],
                        ),
                        child: IconButton(
                          onPressed: _playAudio,
                          icon: Icon(_isPlaying
                              ? Icons.pause_rounded
                              : Icons.play_arrow_rounded),
                          iconSize: 40,
                          color: Colors.white,
                        ),
                      ),
                      // Next chapter
                      IconButton(
                        onPressed: widget.chapterIndex <
                                widget.book.chapters.length - 1
                            ? () {
                                Navigator.pushReplacement(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => ChapterReaderScreen(
                                      book: widget.book,
                                      chapterIndex: widget.chapterIndex + 1,
                                    ),
                                  ),
                                );
                              }
                            : null,
                        icon: const Icon(Icons.skip_next_rounded),
                        iconSize: 36,
                        color: widget.chapterIndex <
                                widget.book.chapters.length - 1
                            ? Colors.orange[700]
                            : Colors.grey[400],
                      ),
                    ],
                  ),
                ] else ...[
                  // Generate button
                  SizedBox(
                    width: double.infinity,
                    height: 52,
                    child: ElevatedButton.icon(
                      onPressed: _generateChapterAudio,
                      icon: const Icon(Icons.headphones, size: 24),
                      label: const Text('Listen to Chapter',
                          style: TextStyle(
                              fontSize: 16, fontWeight: FontWeight.w600)),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
