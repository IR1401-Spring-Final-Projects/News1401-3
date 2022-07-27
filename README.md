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
        <a> این لینک </a>
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
python store_models.py 
```

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        حال، به سراغ نصب پکیج‌های مورد استفاده در این برنامه می‌رویم. برای انجام این کار، کافی است دستور زیر را اجرا کنید.
    </p>
</div>

```shell
pip install -r requirements.txt
```

<div dir="auto" align="justify">
    <p style='direction:rtl;text-align:justify;'>
        در اولین اجرای برنامه، لازم است حتماً وی‌پی‌ان خود را وصل کرده و نیم گیگابایت دانلود اضافی در نظر داشته باشید. در اجراهای بعدی، رعایت چنین مواردی نیاز نیست.
    </p>
</div>





You can find the saved
models [here](https://drive.google.com/drive/folders/1j9J7NPYL1h0Bzc_n-yIPsHjliPS11-rb?usp=sharing).