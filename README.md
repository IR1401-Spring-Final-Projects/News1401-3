<div dir="auto" align="center">
    <h3>
        بسم الله الرحمن الرحیم
    </h3>
    <br>
    <h1>
        <strong>
            بازیابی پیشرفته اطلاعات
        </strong>
    </h1>
    <h2>
        <strong>
            موتور جست و جوی اخبار
        </strong>
    </h2>
    <br>
    <h3>
        محمد هجری - ٩٨١٠٦١٥٦
        <br><br>
        ارشان دلیلی - ٩٨١٠٥٧٥١
        <br><br>
        سروش جهان‌زاد - ٩٨١٠٠٣٨٩
    </h3>
    <br>
</div>

---

<div>
    <h3 style='direction:rtl;text-align:justify;'>
        راه اندازی موتور جست و جوی اخبار
    </h3>
</div>

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        در ابتدا، لازم است مدل‌های مورد نیاز برای اجرای برنامه را از 
        <a href="https://www.dropbox.com/sh/rgeut39nqxy4ydv/AAAsrNBISlAVcJb-DxRjw2nia?dl=0"> این لینک </a>
        دانلود کنید و در پوشه اصلی برنامه قرار دهید.
    </p>
</div>

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        ترمینال را در محل پوشه اصلی برنامه باز کنید و فایل store_models.py را از طریق دستور زیر اجرا کنید تا مدل‌های مورد نیاز در محل مورد نظر قرار بگیرند. می‌توانید بعد از انجام این قسمت، فایل zip مدل‌ها را حذف کنید تا حافظه کمتری از دستگاه شما اشغال شود.
    </p>
</div>

```shell
python store_models.py 
```

---

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        با توجه به این که در این برنامه از موتور جست و جوی Elasticsearch نیز استفاده شده است، لازم است آن را به صورت نصب شده بر روی دستگاه خود داشته باشید. جهت نصب و راه اندازی اولیه‌ی Elasticsearch از 
        <a href="https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html"> این لینک </a>
        استفاده کنید.
    </p>
</div>

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        در این مرحله، لازم است به فولدر bin پوشه‌ی محل نصب Elasticsearch بروید و elasticsearch را اجرا کنید. اگر برای اولین بار این دستور را اجرا می‌کنید، ممکن است درگیر نام کاربری و رمز عبور شوید. در این صورت، حتما دستور زیر را اجرا کنید. 
    </p>
</div>

```shell
python set_elasticsearch_credentials.py --username <USERNAME> --password <PASSWORD>
```

---

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        حال، به سراغ نصب پکیج‌های مورد استفاده در این برنامه می‌رویم. برای انجام این کار، کافی است دستور زیر را اجرا کنید.
    </p>
</div>

```shell
pip install -r requirements.txt
```

---

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        در اولین اجرای برنامه، نیم گیگابایت دانلود اضافی در نظر داشته باشید. البته، این مورد در دفعات بعد لازم نیست؛ ولی توجه کنید که در هر اجرا، لازم است حتماً وی‌پی‌ان شما متصل باشد.
    </p>
</div>

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        در نهایت، دستورات زیر را برای شروع برنامه اجرا کنید.
    </p>
</div>

```shell
cd UI/webui
python manage.py runserver
```

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
       اکنون می‌توانید با باز کردن مرورگر و رفتن به آدرس <a href="http://localhost:8000"> localhost:8000 </a> از رابط کاربری استفاده کنید. 
    </p>
</div>


---

<div>
    <h3 style='direction:rtl;text-align:justify;'>
        پوشه‌ها، پرونده‌ها و کلاس‌ها
    </h3>
</div>

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        تمام نوتبوک‌های تمرین‌های ۳ تا ۵ و هم‌چنین نوتبوک‌های Elastic Search, Query Expansion و هم‌چنین نسخه بهبودیافته دسته‌بندی در پوشه اصلی قرار دارد. رابط کاربری و تمام داده‌های مورد نیاز آن در پوشه UI قرار دارد. رابط کاربری بر اساس فریم‌ورک django ساخته شده است و از bootstrap برای بخش فرانت آن استفاده شده است.
    </p>
    <p style='direction:rtl;text-align:justify;'>
     در پوشه UI یک پروژه django به نام webui ساخته شده است. بر روی این سرور، یک اپلیکیشن mir ساخته شده است که برنامه اصلی ما برای اجرا شدن به روی سرور django می‌باشد.  
    </p>
    <p style='direction:rtl;text-align:justify;'>
        در این پوشه، فایل views.py وجود دارد که وظیفه مدیریت ارتباط فرانت و مدل‌های بازیابی متفاوت از قبیل جست‌و‌جو با مدل‌های Transformers, FastText, TF-IDF, Boolean و مدل ElasticSearch، دسته‌بندی با مدل‌های Logistic Regression و Transformers و خوشه‌بندی با مدل KMeans را انجام می‌دهد.
    </p>
     <p style='direction:rtl;text-align:justify;'>
        در فایل model_classes.py کلاس مدل‌های بازیابی قرار دارند که نشان‌دهنده منطق بازیابی هر کدام از مدل‌ها هستند. هم‌چنین در این فایل رابط کاربری برای برقراری ارتباط با Elastic Search قرار داده شده است. تعدادی از ویژگی‌های ثابت مورد نیاز مدل‌ها (مانند تبدیل کد کلاس هر دسته به نام آن دسته در مسئله دسته‌بندی) نیز در این فایل قرار دارد.
    </p>
    <p style='direction:rtl;text-align:justify;'>
        در فایل urls.py نحوه ارتباط برنامه با سرور اصلی django مشخص شده است و هم‌چنین، در فایل Elastic_Credentials.json نام کاربری و رمز عبور جهت اتصال به سرور مشخص شده است. توجه داشته باشید در صورتی که این رابط را بر روی کامپیوتر خود اجرا می‌کنید، حتماً نام کاربری و رمز عبور را مطابق با Elastic Search سیستم خود تغییر دهید.
    </p>
    <p style='direction:rtl;text-align:justify;'>
        در پوشه models تمام مدل‌ها و داده‌های مورد نیاز برای انجام فرایندهای جست‌و‌جو، دسته‌بندی، خوشه‌بندی و تحلیل لینک وجود دارد. توجه داشته‌باشید در صورتی که این برنامه را  بر روی کامپیوتر خود اجرا می‌کنید، حتماً فایل‌های مورد نیاز را از لینکی که در بالا در اختیار شما قرار داده شده است دریافت کرده و در این پوشه قرار دهید
    </p>
</div>

---

<div>
    <h3 style='direction:rtl;text-align:justify;'>
        ارزیابی MRR
    </h3>
</div>

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        برای ارزیابی کارایی موتور جست و جوی اخبار کافی است به فایل MRR_measure.xlsx مراجعه کنید.
    </p>
</div>
