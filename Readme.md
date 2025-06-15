# 大模型路由策略实验复现要求

# 实验交付要求

实验交付需求：做的各个实验都需要提供日志比如打出log这样来进行证明，类似：https://github.com/anyscale/llm-router/blob/main/README.ipynb里面的过程，特别是论文里提及的六个实验，一定要交付出实验日志（类似复现实验的log供检查）。

## 实验要求概述

大语言模型路由策略模型实现，目标是根据用户输入问题的难易程度动态选择难一点的模型如（DEEEPSEEK 32B）或弱一点的模型如（LLAMA 8B）进行回答从而实现大语言模型路由策略。我设计的RAGRouter算法跟现有的大语言模型路由策略RouterLLM的五个路由策略进行了比照，并可以验证性能得到了一定的提升。实验一共要做六个实验，这六个实验可以分为对比实验跟消融实验，实验一到实验二为RAGRouter模型于RouteLLM模型的对比实验，实验三到实验六为RouteLLM自身的消融实验，实验的测量指标都为CPT（50%），CPT（80%），APGR。实验一为将RAGRouter跟五个路由策略在MMLU测试数据集上进行比较；实验二为将RAGRouter跟五个路由策略在GSM8K 上进行比较；实验三为多來源資料融合實現資料增強與否實驗對比，主要作用为测量是否对训练数据使用文中的两个方法进行数据增强对实验的影响，所以消融模型为只使用原始训练数据集进行训练的模型。实验四为資料採樣與否實驗對比，主要作用为测量通過基於困惑度的動態採樣策略去提高訓練集中高品質用戶請求樣本的比例与否对实验的影响。实验五为語義句法特徵提取與否實驗對比，主要作用为采用跨模态多头注意力机制提取训练数据集的语义句法特征对实验的影响。实验六为搭建 RAG 框架與否實驗對比，主要作用为在路由策略模型训练中搭建RAG 框架通過動態檢索與使用者請求問題相似的實例及其標籤从而構建增強的輸入提示与否对实验的影响。要求能跑出表格里的数据的数据，由于大模型实验数据可能存在一定的波动，但要求不要和论文中表格内的数据相差太大，还有实验一到实验二要求RAGRouter的指标要比RouteLLM的要好，即CPT（50%）+CPT（80%）要比RouteLLM的低，APGR要比RouteLLM的高，消融实验实验三到六要求做了消融的实验效果要比没做消融的差，即做了相应模块消融的模型CPT（50%）+CPT（80%）要比没做消融的高，APGR要比没做消融的低。

## RAGRouter模型介绍

RAGRouter的具体设计原理如下：

選擇 RAGRouter 模型作為路由策略的覈心框架，是因為其能够通過融合語義與句法特徵並結合檢索增强生成科技，實現對査詢問題難易程度的精准評估與動態模型選擇。

​        在進行模型的正式訓練前，我們對訓練所採用的數据集進行折開，基於各模塊在模型訓練中所承擔的功能與數據需求，本文將訓練數據劃分為兩部分：約 70%用於構建RAG 向量庫，主要用於覆蓋盡可能多樣化的語義和句法特徵，從而在相似性檢索中提供豐富且有效的候選數據，提升檢索匹配的精度與魯棒性；剩餘約 30%則作為提示輸入數據，用於微調路由策略模型，以確保模型能够準確捕捉用戶査詢的難易特徵並進行合理匹配。該 7:3 的折開策略在兼顧向量庫多樣性與規模的同時，也保證了微調數據的質量與均衡性，從而有效避免過擬合併提升模型的泛化能力，後續可通過交叉驗證等方法進一步優化這一數據比例。在模型的配寘方面，我們採取了以下步驟：

​       （1）語義特徵選取：對數據集中的請求問題分詞處理以準備後續分析，利用DeBERTa-v3 模型生成上下文感知的語義嵌入矩陣，其中 DeBERTa-v3 是一種先進的預訓練語言模型，利用解耦注意力機制捕捉詞彙間的上下文關係，從而選取深層語義特徵𝐻𝑠𝑒𝑚（维度为 768）。

​         (2)句法特徵選取：通過對査詢問題進行依存語法分析構建句法依存圖以捕捉句子結構，並利用 GraphSAGE 模型對依存圖進行多層特徵傳播增强句法表示，其中GraphSAGE 是一種圖神經網路，專門用來學習圖中節點的嵌入表示，最終選取句法特徵𝐻𝑠𝑦𝑛（維度為 128）。

​        (3)語義與句法特徵融合：對於已選取語義特徵的向量 *Hsem*，已選取句法特徵的向量𝐻𝑠𝑦𝑛，為了將它們的特徵進行融合，設計跨模態多頭注意力機制（CM-MHA）用於實現語義特徵和句法特徵的互動。通過將語義特徵𝐻𝑠𝑒𝑚作为 Query**，**句法特徵𝐻𝑠𝑦𝑛作为Key-Value，用多頭注意力機制捕捉兩者的互動模式，從而生成融合特徵向量後進行平均池化得到向量ℎ𝑓𝑢𝑠𝑒·𝑔𝑙𝑜𝑏𝑎𝑙（維度为 256）。

​       (4)RAG 向量知識庫搭建：將用戶偏好數據集中劃分了搭建 RAG 向量資料庫的數据集對數據集中用戶輸入請求的問題使用 DeBERTa-V3 模型與 GraphSAGE 模型區選取它們的語義特徵與句法特徵並得到特徵向量，然後使用跨模態注意力機制將它們的語義特徵與句法特徵融合在一起生成融合特徵向量，最後將融合的特徵向量進行平均池化得到向量ℎ𝑓𝑢𝑠𝑒·𝑔𝑙𝑜𝑏𝑎𝑙，將這些進行了融合的特徵向量存儲到向量資料庫中。

​      (5)向量相似性檢索：將用戶偏好數據集中劃分為類比用戶輸入請求問題的數據集中對數据集用戶輸入請求的問題選取其語義特徵和句法特徵，構建特徵向量，隨後，利用跨模態注意力機制對語義特征和句法特徵進行融合，最終生成融合特徵向量ℎfuse·global，我們選用 KNN 算灋從向量資料庫中進行相似性檢索，找到資料庫中最接近該請求輸入問題嵌入向量的 K 個向量，然後通過向量索引返回資料庫中匹配出 K 個和輸入的請求問題相關的請求問題以及標籤𝑙𝑖,𝑗（即該用戶請求問題適合强模型還是弱模型回答）。

​     (6)構建增强輸入：RAG 將檢索到的 K 個與輸入請求問題意思相近的問題以及標籤集合起來，形成一個全面的問題提示（Prompt），在構建提示時，將輸入請求問題與檢索到的相似問題及其標籤𝑙𝑖,𝑗結合，這些全面的問題提示有助於增强上下文資訊，優化模型選擇策略。

​     (7)模型微調層：對於輸入的全面問題提示（prompt），我們選擇把它們輸入到QWEN 7B 大模型進行全參數微調，通過更新模型的所有參數來適應分類任務，調整模型的每一個權重，模型的整個參數空間都會根據具體任務的損失函數進行優化。對QWEN 7B 大模型進行全參數微調的具體目標是將其轉換為一個質量評分預測器即預測弱模型對査詢響應質量的打分，用於支持 LLM 路由器的智慧決策。

​    (8)輸出層：最終，全連接層將 QWEN 7B 的隱藏層輸出映射到 5 個類別對應的logits（預測弱模型對査詢響應質量 5 個打分類別的得分），然後通過 softmax 轉換為概率分佈，從而支持後續的決策，為分類任務提供了一個穩健且易於解釋的輸出機制。

  （9）優化器：為了使大模型模型在微調過程能够自我調整調整每個參數，選擇 Adam優化器，保持訓練過程的穩定。

通過精心設計的 RAGRouter 路由策略模型，我們成功融合了語義與句法特徵，並結合檢索增强生成科技（RAG）與全參數微調 QWEN 7B 大模型，實現了對査詢問題難易程度的精准評估與動態模型選擇，這種設計充分利用了語義與句法特徵的互補性，並通過 RAG 科技提升了模型對複雜査詢的理解與處理能力，構建高效的大語言模型路由策略框架。

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/RAGRouter%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E5%9B%BE.png)

## 一：训练数据集的选择

### 1.1训练数据集概括

为了训练大语言模型路由策略，我们选择了Chatbot Arena Human Preference Dataset（拥有55000条数据）跟LLM-judge-labeled datasets （拥有120000条数据），所以在不进行训练数据增强前（即拓增训练数据），一共有120000+55000=175000行数据，下面的内容大概简要介绍一下训练数据集。

### 1.2训练数据集介绍

其中Chatbot Arena Human Preference Dataset平台的数据集约有55000条数据，大概的标记形式是对于输入问题的查询，如果模型强模型即如所选的强模型a获胜则在winner_model_a上标记为1，其它的winner_model_b跟winner_tie标记为0，如果模型弱模型即如所选的弱模型b获胜则在winner_model_b上标记为1，其它的winner_model_a跟winner_tie标记为0。如果强模型a和弱模型b性能都一样，打平手，则winner_tie标记为1，winner_model_a和winner_model_b标记为0。LLM-judge-labeled datasets则也是一个公開的數據集，包含約 120,000 個樣本，每個樣本包括用戶查詢、來自強 LLM（GPT-4）和弱 LLM（Mixtral-8x7B）的響應，以及一個由 GPT-4 評判的 1 到 5 的相對質量評分。大概的标记形式为是对于用户已有输入的查询，有强模型GPT-4作为代表的回答以及弱模型Mixtral-8x7B作为代表的弱模型，其中gpt4_response是LLM-judge-labeled datasets上已有的回复， mixtral_response为Routellm团队选取Mixtral-8x7B用于对用户查询问题prompt的回答，然后进行打分，mixtral_score表示由 GPT-4 評判的弱模型回應的相對品質評分（1-5）其中4-5分则代表弱模型回答质量好，这个查询prompt应当路由到弱模型，1-3分表示弱模型回答质量不够好，应当路由到强模型进行回答

Chatbot Arena Human Preference Dataset 是一個公開的數據集 ，託管在（https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k）上，包含超過 55,000 個真實用戶查詢和來自 64 個最先進 LLM（如 GPT-4、Claude 2、Llama 2、Gemini 和 Mistral 模型）的響應。Chatbot Arena Human Preference Dataset 資料集內容介紹詳細如下：

![](C:\Users\Administrator\Desktop\Large model routing experiment requirements\png\chatbot Arena数据集介绍.jpg)

数据集的具体呈现内容为（截取部分）：

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/chatbot%E5%B9%B3%E5%8F%B0%E6%95%B0%E6%8D%AE%E9%9B%861.jpg)

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/chatbot%E5%B9%B3%E5%8F%B0%E6%95%B0%E6%8D%AE%E9%9B%862.jpg)

LLM-judge-labeled datasets则也是一个公開的數據集，包含約 120,000 個樣本，数据托管在https://huggingface.co/datasets/routellm/gpt4_dataset，数据集的具体呈现内容为（截取部分）：

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/gpt%204%20dataset%E5%9B%BE%E4%B8%80.jpg)

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/gpt4%20dataset%E5%9B%BE%E4%BA%8C.jpg)

## 二：数据增强

对于Chatbot Arena Human Preference Dataset（拥有55000条数据）跟LLM-judge-labeled datasets （拥有120000条数据）这两个原有的数据集，按照RouteLLM的训练思路，训练数据集越多，训练出来的路由策略模型效果越好，所以要对训练数据集进行数据增强，即拓充数据集，参考论文的内容，使用基于依存句法分析进行数据增强还有使用基於知識圖譜引導的語義擴展拓充訓練数据集方法来进行数据增强。大概的思路是从Chatbot Arena跟LLM-judge这两个数据集中抽取部分质量高的数据的用户的查询问题，具体如何判断数据质量的高与否可以通过类似GPT-4等强大模型来进行判断也可以按照其它的技术方法来判断，大概从两个数据集加起来的175000行数据中挑选约50000行数据中的用户查询问题，这50000行数据的用户查询问题prompt里面有些使用依存句法分析进行数据增强有些使用基於知識圖譜引導的語義擴展来进行数据增强后拓充了的数据大约为225700行，拓充后具体的数量也可以参考你们技术人员具体情况来定，但原则就是数据有拓充就好，以及拓充了数据以后后面还要进行数据集采样来平衡数据，拓充完了的数据以及将所有数据进行完采样以后数据量要比原有的175000行数据要多。

对于使用了以上两种数据增强方式对训练数据集的用户查询问题prompt进行数据拓充以后，生成变异查詢𝑞′，把每個查詢𝑞′分別輸入給強模型 Deepseek 32B 與弱模型 LLAMA 8B 生成對應的回答，生成回答後，使用 DeepSeek 671B 大模型進行驗證打分的方式，對弱模型 LLAMNA 8B 生成的回答內容品質進行打分，打分制 5 分制，如果打分為 1-3 分，則標記路由到強模型，如果打分為 4-5 分，則路由到性能弱的模型，以此來為拓充的資料集生成新的偏好標籤1𝑖,𝑗′。



## 三：数据采样

为了解决训练数据集中不同難度樣本分佈不均的問題，，类似比如不会出现简单的查询过多参考论文的内容采用参考论文技术文档的PPL方法，我們採用基於困惑度（Perplexity，PPL）的動態採樣方法，對於每個査詢 *q*，均採用GPT-2-large或相关的大模型計算其困惑度 PPL（*q*）方法进行采样。对于希望数据集可以标签平衡，即属于路由到难的模型的问题或属于路由到简单的模型的问题这些属于查询问题的标签不会出现特别不平衡的现象，参考https://github.com/anyscale/llm-router/blob/main/README.ipynb中Label Rebalancing在In[5]行代码的做法，也是相当于是做了数据采样，他们也是建议在标签平衡的数据集上进行训练，以确保模型不会偏向于特定标签并觉得因此可以提升模型的性能的方法。使用以上两种方法对训练数据集进行采样处理后，类似的原有的训练数据集175000数据加上数据增强的约50000行数据共225000行数据它们一起做了数据采样以后大概采样完的数据有200000行数据，原则就是比原有的175000数据以及比做了数据增强后的数据少就好，还有就是做完数据采样以后确实要出现标签平衡的效果。



## 四：数据预处理

在訓練大語言模型的路由策略模型之前，對訓練數據集進行預處理是至關重要的一步，因為它確保了數據格式的適應性，便於模型學習和優化。详细的可以参考我论文的内容。在数据预处理的过程主要是对训练数据集进行数据标签对齐，主要是对Chatbot Arena Human Preference Dataset 的55000行数据进行操作，該數據集的標籤為 winner_model_a、winner_model_b 和 winner_tie，採用二元形式表示模型比較結果。然而，路由策略模型需要 1-5 分的評分形式。為此，用基於大模型的評分方法，選取 DEEPSEEK 671B 大模型作為評估工具，對弱模型（model_b）的回答品質進行 1-5 分的量化打分。1-3 分：表示回答品質不足，問題適合由弱模型處理；4-5 分：表示回答品質良好，問題更適合由強模型回答。此評分標準充分利用了 DEEPSEEK 671B 模型的語言理解能力，類比人類對回答品質的評估，從而推斷問題的模型匹配度以及確保標籤連續性和適應性，為模型微調奠定基礎，因为后面微调QWEN 7B大模型是需要训练一个 5 路分类器来预测用户查询问题prompt的弱模型得分，所以對弱模型（model_b）的回答品質進行 1-5 分的量化打分也是有必要的这一步处理。



## 五：训练数据用户查询prompt特征提取处理

参考论文里的对用户查询问题prompt通过 DeBERTa-v3 提取语义特征*Hsem*，通过 GraphSAGE 提取句法特征*Hsyn*，并並利用跨模態多頭注意力機制（CM-MHA）生成融合特徵Hfuse，但由于要做RAG向量相似度檢索，所以只能用單一的 256 維度向量計算相似度，對此，我們采取平均池化的方法，對*Hfuse* 的 n 个 256 維向量取平均值，生成單一的 256 維向量ℎ𝑓𝑢𝑠𝑒·𝑔𝑙𝑜𝑏𝑎𝑙。





## 六：RAG向量数据库搭建

每個査詢問題的向量ℎ𝑓𝑢𝑠𝑒·𝑔𝑙𝑜𝑏𝑎𝑙與其對應的路由標籤𝑙𝑖,𝑗（例如“適合模型：GPT-4”或“適合模型：Mistral 7B”）共同存儲，形成知識庫的基本單元。完成特徵向量化後，所有査詢問題及其融合特徵向量被存儲至知識庫中。實現高效的向量檢索，本研究採用 Faiss 庫構建向量索引。Faiss 是一種高效的相似性搜索工具，支持大規模向量的高效檢索，選用 IVF 索引作為基線索引方法，計算歐幾裡得距離（L2 距離）作為相似性度量用于后面的RAG相似性向量检索。



## 七：微调QWEN 7B大模型

微調目標是使 QWEN 7B 預測弱模型（如 QWEN 7B 自身）對輸入查詢響應的質量評分，輸出 5 個類別的概率分佈（對應評分 1-5 分，其中 1-3 分代表弱模型回答質量不好需要該査詢需要由强模型來回答，4-5 分代表弱模型回答質量好，該問題可以由弱模型來回答）。輸入為用户的查询问题Prompt以及通過 RAG 構造的增强提示，輸出用於决定模型選擇。详细的内容参考论文文档内容





## 八：RAGRouter测试：MMLU上与GSM8K上进行

训练出来的路由策略模型RAGRouter要在测试数据集MMLU与GSM8K上进行测试。

MMLU：MMLU 是一個由 Hendrycks 等人在 2020 年提出的基準，旨在測試大型語言模型在多工場景下的理解能力。該資料集包含 14,042 個問題，覆蓋 57 個學科，包括STEM（科學、技術、工程、數學）、人文和社會科學等領域。每個問題均為多選題，模型需從選項中選擇正確答案，性能以準確率衡量。MMLU 的多樣性使其成為評估路由模型在處理廣泛查詢類型時的理想選擇，特別是當查詢涉及不同知識領域時，路由模型需選擇適合的 LLM 以優化回答品質。

GSM8K：GSM8K（Grade School Math 8K）是由 Cobbe 等人在 2021 年提出的資料集，旨在測試模型在小學數學問題上的推理能力。該資料集包含 8,792 個問題，其中訓練集有 7,473 個，測試集有 1,319 個，每個問題均需逐步推理求解，答案為數值或運算式。我們使用其測試集（1,319 個問題）進行評估，性能通過所選 LLM 解題的正確率衡量。GSM8K 特別適合測試路由模型在選擇適合數學推理任務的 LLM 方面的能力 [49]，例如當問題涉及多步計算時，路由模型需選擇更強的 LLM 以確保準確性。

大概通俗易懂说明一下，MMLU测试数据集大概由57个.csv文件组成，每个文件都涵盖不同的学科问题如mmlu_astronomy.csv文件则代表天文学领域的测试数据集，测试数据集的示例如截图所示：

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/mmlu%E7%A4%BA%E4%BE%8B%E5%9B%BE.jpg)

mmlu的测试数据集大概内容就是包含三个列表，分别代表prompt即用户的提示词，强模型gpt-4-1106-preview的回答对与错（回答正确则用True表示，回答错误则用FALSE表示）以及弱模型mistralai/Mixtral-8x7B-Instruct-v0.1的回答对与错（回答正确则用True表示，回答错误则用FALSE表示）。

GSM8k测试数据集由一个.csv文件组成，包含的是数学问题，测试数据集的示例如截图所示：

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/gsm8k%E7%A4%BA%E4%BE%8B%E5%9B%BE.jpg)

GSM8K的测试数据集大概内容就是包含五个列表，分别代表prompt即用户的提示词，强模型gpt-4-1106-preview的回答对与错（回答正确则用True表示，回答错误则用FALSE表示）和所回答的答案response以及弱模型mistralai/Mixtral-8x7B-Instruct-v0.1的回答对与错（回答正确则用True表示，回答错误则用FALSE表示）和所回答的答案response。



## 九：表格一到表格六实验

表格一实验：RAGRouter与RouteLLM在MMLU测试数据集上进行性能比较

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/%E8%A1%A8%E6%A0%BC%E4%B8%80.jpg)



表格二实验：RAGRouter与RouterLLM在GSM8K测试数据集上进行性能比较

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/%E8%A1%A8%E6%A0%BC%E4%BA%8C.jpg)



表格三实验：多來源資料融合實現資料增強與否實驗對比

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/%E8%A1%A8%E6%A0%BC%E4%B8%89.jpg)

表格四实验：資料採樣與否實驗對比

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/%E8%A1%A8%E6%A0%BC%E5%9B%9B.jpg)

表格五实验：語義句法特徵提取與否實驗對比

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/%E8%A1%A8%E6%A0%BC%E4%BA%94.jpg)

表格六实验：搭建 RAG 框架與否實驗對比

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/%E8%A1%A8%E6%A0%BC%E5%85%AD.jpg)



## RouteLLM路由策略说明

RouteLLM的官网地址为：https://github.com/lm-sys/RouteLLM/tree/main，RouteLLM有四个路由策略，分别为Similarity-weighted (SW) ranking，Matrix factorization，BERT classifier，Causal LLM classifier；这四个路由策略也下载了在AutoDL的4090D服务器上，其路由策略已完成下载，可以直接对其进行调用。

对于RouteLLM的代码，我这边在AutoDL的服务器上的RTX 4090D * 1卡上部署了环境，可以实现调用四种路由策略模型，即RouteLLM上的Similarity-weighted (SW) ranking，Matrix factorization，BERT classifier，Causal LLM classifier 这四种路由策略，已经完成下载，但是如果要进行跑出相关测试结果的话，要类似调用routellm.evals.evaluate库，参考https://github.com/anyscale/llm-router/blob/main/README.ipynb的In[13]!python -m routellm.evals.evaluate,按照我理解也就是要调用RouteLLM的代码文件https://github.com/lm-sys/RouteLLM/blob/main/routellm/evals/evaluate.py，具体就看你们技术的操作。然后运行测试数据集跑出CPT（50%），CPT（80%），APGR这三个指标也可以参考上述内容

AUTODL的示例图如下：

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/RouteLLM%E5%9C%A8autodl%E4%B8%8A4090D%E7%9A%84%E9%83%A8%E7%BD%B2.jpg)



## 评测指标说明

实验的结果指标为CPT（50%），CPT（80%），APGR这三个指标，具体的介绍可以参考发的实验介绍参考文档，下面，我将通俗易懂一点介绍CPT跟APGR指标的具体原理

1.CPT指标：

CPT指标全称为呼叫性能閾值，個旨在評估基於路由的系統中為實現預定義性能目標所需的最小計算資源的指標。大概的计算方式可以理解为如下：

首先如果我们在测试数据集GSM8K和MMLU这两个测试数据集上进行测试，测试的指标是我们的路由器在测试的过程中对于每个问题路由了强模型或弱模型，对于每个问题，强模型或弱模型的回答可能表现为对或者错，以GSM8K为例，有1319个测试问题，其中，这1319个问题有的问题让强模型回答其回答结果有正确也有错误，弱模型同样如此；假设这1319个问题都让强模型gpt-4-1106-preview回答，则正确的有1130个，正确率为1130/1319，打分可表示为100×（1130/1319）=85.6分；假设这1319个问题都让弱模型Mixtral-8x7B-Instruct-v0.1回答，正确的有842个，正确率为842/1319，打分可表示为100×（842/1319）=63.8分。

CPT（50%）代表使用了路由策略以后，其打的分数在强模型与弱模型得分的50%中（即：x-63.8/85.6-63.8=50%，解得x为74.55分，所以正确率为74.55%）调用强模型的比例，比如达到此条件调用强模型gpt-4-1106-preview的比例为30.48%即1319个问题中调用强模型gpt-4-1106-preview的次数为1319*30%=396次，则CPT(50%)为30.48%。

同理可得，CPT（80%）代表使用了路由策略以后，其打的分数在强模型与弱模型得分的80%中（即：X-63.8/83.5-63.8=80%，解得X为79.56分，所以正确率为79.56%）调用强模型的比例，比如达到此条件调用强模型gpt-4-1106-preview的比例为41.81%即1319个问题中调用强模型gpt-4-1106-preview的次数为1319*41.81%=551次，则CPT（80%）为41.81%

2.APGR指标
$$
\text{APGR 是 PGR 關於成本函數c}(\mathbb{R}_\alpha^{\mathrm{bin}})\text{的積分,當}\alpha\text{從 0 到 1变化时}
$$

$$
APGR(R_{bin})=\int_0^1PGR(R_\alpha^{bin})d(c(R_\alpha^{bin}))
$$

$$
\text{在實踐中,我們通過將成本區間}[0\%,100\%]\text{離散化為 10 個等距點}\{c_i\}_{i=1}^{10},\text{ 為每個}c_i\text{確定相應的國值}\alpha_i\text{,並計算}:
$$

$$
APGR(R_{bin})\approx\frac{1}{10}\sum_{i=1}^{10}PGR(R_{\alpha_{i}}^{bin})
$$



從幾何上看，APGR 表示 PGR 相較于成本的曲線下面積，提供了一個整體的效率度量。較高的 APGR 表明路由策略能夠在更低的成本下恢復更多的性能差距，有效平衡品質與效率。可以举个以下的例子更好解释说明APGR的计算过程：

APGR的计算过程可以理解为以下过程：

计算：row（AUC）=[Performance（10%）+Performance（20%）+Performance（30%）+Performance（40%）+Performance（50%）+Performance（60%）+Performance（70%）+Performance（80%）+Performance（90%）+Performance（100%）]/10这样的一个值，Performance代表在使用了所对应的路由策略以后在调用对应的强模型比例如perfomance（10%）表示使用了路由策略后调用强模型的比例为10%，perfomance表示性能分数也可以理解成正确率*100，如performance（10%）=（900/1319）×100=68分，900意为正确了900道题，1319可以理解为题目总数。当然在RouteLLM作者给出的代码块中，AUC的计算方法为计算在使用梯形法则计算曲线下面积的方法，也就是积分的方法计算，上述我提到的算式大概是提供一个通俗一点的解释，也就是按照上述的类似对应0%到100%与对应的performance值进行积分进行。

APGR的计算公式为APGR=(row["AUC"] - weak_auc) / (strong_auc - weak_auc)， weak_auc可以理解成假如全部调用弱模型，曲线的面积，换言之，就是假设都调用弱模型，得分是多少，如假设都采用弱模型，1319道题可以对500道题，得分为（700/1319）*100=53分；*strong_auc可以理解成假如全部调用强模型，曲线的面积，换言之，就是假设都调用强模型，得分是多少，如假设都采用强模型，1319道题可以对1000道题，得分为（1000/1319）×100=75分，则APGR=（68-53）/（75-53）=0.68

计算APGR与CPT指标可以参考RouteLLM里的https://github.com/lm-sys/RouteLLM/blob/main/routellm/evals/evaluate.py内容，里面也有详细把测试APGR与CPT的内容用代码进行了表达；同时也可以阅览一下RouteLLM给出的训练路由器模型Causal LLM并在GSM8K测试数据集上进行测试的代码https://github.com/anyscale/llm-router/blob/main/README.ipynb，里面最后要测试出CPT与APGR指标体现在IN[13]即第十三行指令：!python -m routellm.evals.evaluate --config config.example.yaml --routers random causal_llm --benchmark gsm8k也可以进行参考，因为作者也是使用脚本调用了RouteLLM代码文件的evaluate.py跑出相关指标。相关的代码块可以参考evaluate.py的以下内容：

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/CPT%E4%B8%8Eperfomance%E6%8C%87%E6%A0%87%E8%A1%A8%E8%BE%BE.jpg)

![](https://github.com/sky666-glitch/Ragrouter-requirements-recurrent-hope666/blob/master/png/APGR%E4%BB%A3%E7%A0%81%E5%86%85%E5%AE%B9.jpg)

## 实验步骤说明：

实验步骤大的步骤主要分为以下三个大步骤，小的步骤在里面也有具体细分了，先具体跑通RouteLLM的实验，在这过程也学会如何调用evaluate.py函数去测量CPT和APGR这两项指标类似起到这些作用，然后就训练出RAGRouter模型，如果中间过程出了相关的一些问题的话，也可以参考其相关的思想换些方法去训练，保证工作量和思想跟RAGRouter模型训练的过程内容的多少差不多就好，然后后面就做对比实验跟消融实验，实验过程也可以灵活处理，可以看出RAGRouter性能比RouteLLM要好就好。

### （1）：跑通RouteLLM实验

复现RouteLLM的实验，跑通RouteLLM四个路由策略Similarity-weighted (SW) ranking +Matrix factorization+BERT classifier +Causal LLM classifier以及随机路由策略Random模型，在MMLU测试数据集，GSM8K测试数据集上跑出CPT（50%），CPT（80%），APGR这三项指标，参考论文内容：RouteLLM: Learning to Route LLMs with Preference Data

### （2）：训练RAGRouter路由策略模型

1.使用基於依存句法分析與圖特徵拓展的多結構資料增強方法这两种方法对原有的训练数据集Chatbot Arena Human Preference Dataset和LLM-judge-labeled datasets进行数据增强将加起来大概175000行数据拓充到二十多万行数据。

2.採用基於困惑度（Perplexity，PPL）的動態採樣方法以及标签平衡的方法对原有的数据及做了数据增强加起来的训练数据做采样处理，做了数据采样以后大概采样完的数据可以参考有200000行数据，大概比原有的数据多就好。

3.对训练数据进行训练前预处理，主要是对Chatbot Arena的55000行数据做标签统一的操作，对于里面弱模型回答的评分做1-5分的打分作为标签，即對弱模型（model_b）的回答品質進行 1-5 分的量化打分。

4.对训练数据用户查询prompt特征提取处理，通过 DeBERTa-v3 提取语义特征*Hsem*，通过 GraphSAGE 提取句法特征*Hsyn*，并並利用跨模態多頭注意力機制（CM-MHA）生成融合特徵。

5.搭建RAG向量知识库，向量知识库的内容为每個査詢問題的向量ℎ𝑓𝑢𝑠𝑒·𝑔𝑙𝑜𝑏𝑎𝑙與其對應的路由標籤𝑙𝑖,𝑗（例如“適合模型：GPT-4”或“適合模型：Mistral 7B”）共同存儲，形成知識庫的基本單元。

6.微调QWEN 7B大模型（或者类似性能的大模型），輸入為用户的查询问题Prompt以及通過 RAG 構造的增强提示，輸出用於决定模型選擇。

### （3）做对比实验和消融实验

1.对比实验：在MMLU测试数据集与GSM8K测试数据集上，将RAGRouter与RouteLLM的四个路由策略加上随机路由策略（Random）进行比较，跑出CPT（50%），CPT（80%），APGR这三项实验指标。

2.消融实验：在MMLU测试数据集上，做四项消融实验，分别为多來源資料融合實現資料增強與否實驗對比和資料採樣與否實驗對比和語義句法特徵提取與否實驗對比以及搭建 RAG 框架與否實驗對比这四项消融实验。

注：我这边的实验交付需求是做的各个实验都需要提供日志比如打出log这样来进行证明，类似：https://github.com/anyscale/llm-router/blob/main/README.ipynb里面的过程，特别是论文里提及的六个实验，一定要交付出日志（类似复现实验的log供检查）。