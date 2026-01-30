## 混元翻译1.5
模型地址：https://modelscope.cn/models/Tencent-Hunyuan/HY-MT1.5-1.8B    

### 转onnx说明
导出onnx参考`python/export_llm`,应用补丁后再导出。由于混元翻译要求特定版本transformers库运行，导出onnx的环境也需按其要求    
```SHELL
pip install transformers==4.56.0
```

### 转rknn说明
rknn3-model-zoo中的c demo代码，在`cpp/main.cc`中已按照混元1.5模型配置    
如果需要使用rknn3-server,由于开源模型中混元翻译1.5的chat模板未针对翻译模型更新，因此，在转rknn前需要修改开源模型中的`chat_template.jinja`如下：    
```
{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}<｜hy_begin▁of▁sentence｜>{{ system_message }}<｜hy_place▁holder▁no▁3｜>{% else %}{% set loop_messages = messages %}<｜hy_begin▁of▁sentence｜>{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}<｜hy_begin▁of▁sentence｜><｜hy_User｜>{{ message['content'] }}{% elif message['role'] == 'assistant' %}<｜hy_place▁holder▁no▁8｜>{{ message['content'] }}<｜hy_place▁holder▁no▁2｜>{% endif %}{% endfor %}{% if add_generation_prompt %}<｜hy_place▁holder▁no▁8｜>{% else %}<｜hy_place▁holder▁no▁8｜>{% endif %}
```

### 使用
编译c demo后，按提示的参数输入即可。   
其中prompt需要参考官方README中说明配置    
#### Prompt Template for ZH<=>XX Translation.
---
```
将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```
---

#### Prompt Template for XX<=>XX Translation, excluding ZH<=>XX.
---
```
Translate the following segment into {target_language}, without additional explanation.

{source_text}
```
---

