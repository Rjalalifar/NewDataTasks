import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import pandas as pd


nltk.download('punkt')
nltk.download('stopwords')

english_stopwords = set(stopwords.words('english'))

persian_stopwords = [
    'و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'برای', 'آن', 'یک', 'های', 'ان', 'یا', 'تا', 'وی', 'نیز', 'اما',
    'است', 'بر', 'هم', 'همچنین', 'دیگر', 'ما', 'شما', 'آنها', 'می', 'شود', 'خود', 'شد', 'توسط', 'او', 'درباره', 'باید',
    'بود', 'هر', 'چه', 'مورد', 'باشد', 'کرد', 'اگر', 'هستند', 'دارد', 'کند', 'خواهد', 'همه', 'یکی', 'پیش', 'من', 'میان',
    'پس', 'نظر', 'داد', 'روی', 'یکدیگر', 'بیشتر', 'هایی', 'نمی‌شود', 'همین', 'شاید', 'هیچ', 'چند', 'کنید', 'دو', 'چهار',
    'پنج', 'شش', 'هفت', 'هشت', 'نه', 'ده', 'صورت', 'زیرا', 'سوی', 'وقتی', 'یعنی', 'حالا', 'نسبت', 'نا', 'تر', 'بسیار',
    'کم', 'رو', 'بعد', 'زیر', 'درمیان', 'روز', 'شب', 'فقط', 'حتی', 'باره', 'سال', 'همچنان', 'درست', 'بیرون', 'خارج',
    'دیروقت', 'مخصوصا', 'درون', 'سراسر', 'آنچه', 'یکدیگر',
]


# persian_text ="""

# سفر، از دوران‌های کهن‌تر تا عصر مدرن، همواره جزء تجربیات بشری مهم بوده است. از آن زمان که انسان نخستین قدم‌های خود را در دنیا گذاشت، مفهوم سفر تغییر و تحول زیادی پیدا کرده است. در قرون گذشته، انسان‌ها به دنبال اکتشافات جغرافیایی جدید و تجارت از راه جاده‌ها و دریاها می‌پرداختند، در حالی که در قرن بیست و یکم، تکنولوژی و ارتباطات جهانی ما را به دنیایی از سفرهای دیجیتالی و اطلاعاتی منتقل کرده است.

# سفر در قرن بیست و یکم یک معنای جدید پیدا کرده است. در این دوران مدرن، ما دیگر به جستجوهای دوران کاوشگران قدیمی نیاز نداریم تا به دنیاهای ناشناخته برودیم. توسط قدرت حمل و ارتباطات مدرن، جهان به پیشانی ما فراخوانده است. به نوعی، همه ما کاوشگران جدیدی هستیم که به جهان به دنبال معانی و تازگی‌ها می‌رود.

# در دوران مدرن، سفر یک تجربه شخصی و چند وجهی شده است. این یک سفر جسمی به عمق جنگل‌های بارانی آمازون می‌تواند باشد، جایی که از خود شگفتی‌های طبیعی، فرهنگ‌های بومی و تلاش‌های حفاظتی برخوردار است. در عین حال، ممکن است این یک سفر ذهنی و ادراری به عمق دانش و تجربه باشد، جایی که به وسیله اینترنت ما از دسترسی به دانش انسانی در هر زمینه‌ای برخورداریم. ما می‌توانیم دوره‌های آنلاین بگذاریم، با افراد متخصص از سراسر جهان همکاری کنیم و در اعماق دانش مطالعه کنیم.

# سفر دیجیتالی و فیزیکی همچنین ترکیب‌های جالبی از تجربه را ممکن می‌کند. فناوری‌های مانند واقعیت مجازی (VR) و واقعیت افزوده (AR) به ما این امکان را می‌دهند که در دنیاهای جدیدی فرار کنیم یا جایی را که هیچ وقت قادر به رفتن به آن نخواهیم بود، تجربه کنیم. از دیدگاه ما، مرز بین دنیای واقعی و دنیای مجازی ناپدید می‌شود و یک عالمه فضای بی‌پایان برای کاوش ایجاد می‌کند.
# """

persian_text ="""
A Journey of Discovery: Exploring the World in the 21st Century
In the 21st century, the concept of a journey has taken on an entirely new meaning. The advent of technology, the globalization of cultures, and the increasing accessibility to remote corners of the world have transformed the way we embark on, experience, and document our journeys. Unlike the explorers of old who set sail into the unknown, we now have the world at our fingertips, thanks to the power of modern transportation and communication. This modern era has given rise to a new breed of travelers, eager to explore both the physical and digital realms, pushing the boundaries of discovery in every conceivable way.
Journeys have been an integral part of human existence since time immemorial. From the nomadic tribes of our ancient ancestors to the Silk Road traders and the pioneering expeditions of Christopher Columbus, journeys have shaped our history, our cultures, and our very understanding of the world. Yet, as we find ourselves in the 21st century, the nature of journeys has evolved dramatically.
At the heart of the contemporary journey lies the desire for exploration, discovery, and self-discovery. Whether it's a physical trek through a remote rainforest, an intellectual quest to decode the mysteries of the cosmos, or a spiritual pilgrimage to find inner peace, the 21st-century journey is deeply personal and multifaceted. It is a quest for meaning, a search for novelty, and an adventure in the broadest sense of the term.
One of the most remarkable aspects of the modern journey is the accessibility of destinations that were once distant dreams. Air travel, high-speed trains, and advanced road networks have made it possible for anyone with a sense of wanderlust to traverse vast distances in a matter of hours or days. Long-haul flights have transformed the world into a global village, making far-flung corners of the Earth as reachable as the local grocery store. The world has become a playground for travelers, offering an endless array of destinations to explore.
Take, for instance, the journey to the depths of the Amazon rainforest. Once a perilous undertaking reserved for the most intrepid explorers, it is now a relatively straightforward endeavor for the modern adventurer. Guided by experienced local guides and armed with state-of-the-art equipment, travelers can venture deep into the heart of this lush and mysterious wilderness. They can witness the unparalleled biodiversity, come face to face with indigenous cultures, and support conservation efforts to protect this fragile ecosystem.
Similarly, the journey to Antarctica has evolved from a perilous expedition into an attainable destination for those seeking the ultimate adventure. Tourist cruises equipped with cutting-edge technology enable travelers to explore the frozen continent comfortably and responsibly. This journey, with its awe-inspiring landscapes and pristine natural beauty, exemplifies the 21st-century traveler's aspirations to experience the world's most remote and unspoiled places.
As we venture forth into the 21st century, our journeys increasingly transcend the physical realm and extend into the digital domain. The internet has opened up an entirely new frontier for exploration, one that is boundless, ever-changing, and deeply interconnected. The journey into cyberspace is an exploration of knowledge, connectivity, and the human experience.
In the digital age, we embark on intellectual journeys that allow us to traverse the vast landscape of information and ideas. The internet provides us with a gateway to the accumulated knowledge of humanity, accessible at our fingertips. We can engage in online courses, collaborate with experts from around the world, and dive into the depths of knowledge, learning about topics that were once confined to the realms of academia.
The digital journey also extends to the world of virtual reality (VR) and augmented reality (AR). These technologies enable us to immerse ourselves in entirely new and simulated environments. With a VR headset, we can travel to distant galaxies, explore the wonders of ancient civilizations, or experience the thrill of scaling the world's tallest mountains, all from the comfort of our living rooms. The boundaries between the real and the virtual blur, creating a dynamic and limitless realm for exploration.
Moreover, social media and online communities have revolutionized the way we share our journeys. Through platforms like Instagram, Facebook, and YouTube, we can document our travels, share our experiences, and connect with a global audience. Our journeys become not just personal quests but also sources of inspiration and information for others. Social media has turned travelers into storytellers, allowing them to weave narratives of their adventures and showcase the beauty, diversity, and challenges of the world.
The digital realm also facilitates a journey of self-discovery. Personal blogs and vlogs, as well as social media platforms, have given individuals a space to share their personal development journeys. Whether it's a quest for self-improvement, a journey of self-acceptance, or an exploration of one's passions and talents, the internet provides a platform for individuals to connect with like-minded people, seek guidance, and chart their paths to personal growth.
The fusion of physical and digital journeys has given rise to the phenomenon of "digital nomadism." Digital nomads are individuals who leverage technology to work remotely while traveling the world. They can explore new destinations, immerse themselves in different cultures, and maintain their careers simultaneously. This lifestyle is a testament to the transformative power of the 21st-century journey, where work and leisure seamlessly blend, and the world becomes one's office.
In addition to the physical and digital dimensions of journeys, there is a significant spiritual aspect to contemporary exploration. Spiritual journeys, in particular, have seen a resurgence in recent years, as individuals seek inner peace, enlightenment, and a deeper connection with the cosmos. These journeys can take many forms, from traditional pilgrimages to modern wellness retreats.
One such spiritual journey is the Camino de Santiago in Spain. This ancient pilgrimage route has drawn people from all walks of life for centuries, with the goal of reaching the city of Santiago de Compostela and paying homage to the relics of Saint James. The Camino is not just a physical trek but a deeply personal and spiritual quest, where travelers often experience profound moments of self-discovery and reflection.
Wellness retreats have also become a popular form of spiritual journey. These retreats offer individuals the opportunity to escape the hustle and bustle of everyday life and immerse themselves in practices that promote physical and mental well-being. From yoga retreats in the Himalayas to meditation retreats in the tranquil settings of Southeast Asia, these journeys are a way to seek balance, healing, and a deeper connection with oneself.
At the heart of every journey, regardless of its nature, lies the element of discovery. In the 21st century, our understanding of the world is constantly evolving, and journeys serve as a means of exploration and education. When we embark on a journey, we learn about new cultures, witness the wonders of nature, and gain insights into the human condition.
Cultural exchange is a vital aspect of modern journeys. Travelers have the opportunity to engage with people from diverse backgrounds, learn about their traditions, and immerse themselves in their daily lives. This cultural exchange fosters mutual understanding and appreciation, breaking down barriers and promoting global unity.
Moreover, journeys often provide firsthand encounters with the natural world. From the towering peaks of the Himalayas to the serene beauty of the Maldives' coral reefs, travelers bear witness to the Earth's breathtaking landscapes and ecosystems. Such experiences inspire environmental awareness and a deeper commitment to conservation, as we come to understand the fragility of our planet and the urgent need to protect it.

"""

# Tokenize text
sentences = nltk.sent_tokenize(persian_text)
words = [nltk.word_tokenize(sentence) for sentence in sentences]

# Language_Variable="Persian"

Language_Variable="English"

# Remove stop words
filtered_words = []
for word_list in words:
    if Language_Variable=="Persian":
        filtered_words.append([word for word in word_list if word not in persian_stopwords and word.isalpha()])
    elif Language_Variable=="English":
        filtered_words.append([word for word in word_list if word.lower() not in english_stopwords and word.isalpha()])

# Calculate word frequency
word_freq = FreqDist(np.concatenate(filtered_words))

# Create vectors
sentence_vectors = []
max_vector_length = max(len(sentence) for sentence in filtered_words)

for sentence in filtered_words:
    vector = [word_freq[word] for word in sentence] + [0] * (max_vector_length - len(sentence))
    sentence_vectors.append(vector)

# Create a graph  similarity
def sentence_similarity(sent1, sent2):
    return 1 - cosine_distance(sent1, sent2)

sentence_graph = nx.Graph()

for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        similarity = sentence_similarity(sentence_vectors[i], sentence_vectors[j])
        sentence_graph.add_edge(i, j, weight=similarity)

# PageRank
scores = nx.pagerank(sentence_graph)

# number of summery sentences
num_sentences_in_summary = 5

# Extract the sentences
sorted_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# final summary
summary_sentences = sorted_sentences[:num_sentences_in_summary]

summary = " ".join([sentences[i] for i, _ in summary_sentences])

data = {"Sentence": [sentences[i] for i, _ in sorted_sentences],
        "Ranking": [scores[i] for i, _ in sorted_sentences]}
df = pd.DataFrame(data)

print(df)
print("\nSummary:")
print(summary)
