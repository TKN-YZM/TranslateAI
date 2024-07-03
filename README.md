# NLP: İngilizce'den Türkçe'ye Çeviri

Bu proje, doğal dil işleme (NLP) alanında çalışan küçük bir yapay zeka modelidir. İngilizce metinleri Türkçe'ye çevirme görevini üstlenir.

## Başlangıç

Bu projenin amacı, GRU (Gated Recurrent Unit) gibi bir model kullanarak dil modellemesi yapmak ve bu modeli İngilizce metinlerini Türkçe'ye çevirmek için kullanmaktır.

## Oluşturma Süreci
Veri Setinin Dahil Edilmesi: İngilizce ve Türkçe metin verileri data/ klasörüne yerleştirilmiş ve proje için uygun formatta düzenlenmiştir.

Tokenizer İşlemleri: tokenizer.py scripti kullanılarak metinler, kelime düzeyinde parçalara ayrılmış ve her bir kelime bir indekse atanmıştır.

Padding İşlemleri: Veriler, eşit uzunluktaki diziler halinde olacak şekilde pad_sequences fonksiyonu ile işlenmiştir.

Encoding ve Decoding İşlemleri: İngilizce metinleri Türkçe'ye çevirmek için GRU (Gated Recurrent Unit) modeli kullanılmıştır. Bu model, metinlerin doğru bir şekilde kodlanıp çözülmesini sağlar.

