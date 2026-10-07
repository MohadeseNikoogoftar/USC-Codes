# Hybrid Image Filtering

این پروژه پیاده‌سازی فیلترهای **پایین‌گذر (Low-Pass)** و **بالاگذر (High-Pass)** و ترکیب آن‌ها برای ایجاد یک **تصویر هیبریدی (Hybrid Image)** است.


اجرای برنامه:

```bash
python hybrid_image.py image1.jpg image2.jpg
```

## Requirements

برای اجرای پروژه به کتابخانه‌های زیر نیاز است:

```text
numpy
Pillow
```

نصب وابستگی‌ها:

```bash
pip install numpy pillow
```

---

## Summary

این پروژه نشان می‌دهد که چگونه می‌توان با استفاده از فیلترهای Gaussian و عملیات Convolution، مؤلفه‌های فرکانسی مختلف یک تصویر را استخراج و سپس اطلاعات فرکانس پایین یک تصویر را با اطلاعات فرکانس بالای تصویر دیگر ترکیب کرد.


