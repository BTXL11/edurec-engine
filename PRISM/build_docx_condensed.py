# -*- coding: utf-8 -*-
"""
Build a 2-3 page condensed version of 汇报8.docx while keeping the original
design: same styles (WeeklyTitle/H1/H2/Caption/Normal), same page setup,
same footer, same table look (copied tblPr + header fill E8EEF0, 9pt bold).
"""
import copy
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

SRC = r"E:\work\learn\PRISM\汇报8.docx"
DST = r"E:\work\learn\PRISM\汇报8_精简版.docx"

BODY_INDENT = Cm(0.74)
LINE = 1.5


def clear_body(doc):
    body = doc.element.body
    kept = []
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            kept.append(child)
        else:
            body.remove(child)
    return kept


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="Normal")
    pf = p.paragraph_format
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.first_line_indent = BODY_INDENT
    pf.line_spacing = LINE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
    p.add_run(text)
    return p


def add_heading(doc, text, level=2):
    style = "WeeklyH1" if level == 1 else "WeeklyH2"
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = Cm(0.85) if level == 1 else Cm(0.78)
    pf.line_spacing = LINE
    pf.space_before = Pt(5)
    pf.space_after = Pt(2)
    pf.keep_with_next = True          # never leave a heading alone at page foot
    p.add_run(text)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph(style="WeeklyCaption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(4)
    pf.space_after = Pt(2)
    pf.keep_with_next = True          # caption stays with its table
    p.add_run(text)
    return p


def set_fixed_widths(t, widths):
    """Force a fixed layout: tblLayout=fixed, tblW, tblGrid and every tcW."""
    tbl = t._tbl
    tblPr = tbl.tblPr
    for tag in ("w:tblLayout", "w:tblW"):
        el = tblPr.find(qn(tag))
        if el is not None:
            tblPr.remove(el)
    total = sum(int(w.twips) for w in widths)
    tblW = tblPr.makeelement(qn("w:tblW"), {qn("w:w"): str(total), qn("w:type"): "dxa"})
    tblPr.append(tblW)
    layout = tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
    tblPr.append(layout)

    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        tbl.remove(grid)
    grid = tbl.makeelement(qn("w:tblGrid"), {})
    for w in widths:
        grid.append(grid.makeelement(qn("w:gridCol"), {qn("w:w"): str(int(w.twips))}))
    tblPr.addnext(grid)

    for row in t.rows:
        for i, cell in enumerate(row.cells):
            cell.width = widths[i]


def add_table(doc, ref_tblpr, rows, widths, header=True, accent_rows=()):
    n_rows, n_cols = len(rows), len(rows[0])
    t = doc.add_table(rows=n_rows, cols=n_cols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # reuse the original table properties (style id 34, borders, layout)
    old = t._tbl.tblPr
    t._tbl.replace(old, copy.deepcopy(ref_tblpr))
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = t.cell(r_i, c_i)
            cell.width = widths[c_i]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(val))
            run.font.size = Pt(9)
            if (header and r_i == 0) or (r_i in accent_rows):
                run.bold = True
            if header and r_i == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = tcPr.makeelement(qn("w:shd"), {
                    qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): "E8EEF0"})
                tcPr.append(shd)
    # enforce column widths on every cell
    set_fixed_widths(t, widths)
    return t


def main():
    doc = Document(SRC)
    # remember the look of a data table before wiping the body
    ref_tblpr = copy.deepcopy(doc.tables[5]._tbl.tblPr)
    clear_body(doc)
    # drop the image relationships as well, otherwise the ~2 MB of now-unused
    # media stays inside the package even though nothing references it
    for rId in [r for r, rel in doc.part.rels.items() if "image" in rel.reltype]:
        doc.part.drop_rel(rId)

    # ---- title
    p = doc.add_paragraph(style="WeeklyTitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0.99)
    pf.line_spacing = 1.5
    pf.space_before = Pt(3)
    pf.space_after = Pt(6)
    p.add_run("进展报告")

    add_body(doc, "本周围绕《PRISM: Personalized Recommendation via Information Synergy Module》"
                  "（WWW 2026）开展研读与复现：完成 Amazon Beauty 数据准备与真实 CLIP 图文特征生成，"
                  "修正开源代码的 faiss 导入与 λ 参数问题，建立纯 ID、图文直加、PRISM 三类对照，"
                  "并用三个随机种子复核。核心结论是：PRISM 在本地带来稳定的正向提升，但幅度明显小于论文，"
                  "差距主要源于基线强度与实现细节。")

    # ---- 一
    add_heading(doc, "一、具体工作内容", 1)

    add_heading(doc, "1. 问题定位与选题依据", 2)
    add_body(doc, "前期接入 edurec-engine 后，用采集的 B 站数学建模资源测试，尚未观察到明显优于直接检索的效果。"
                  "除标题冗长外，简介中的互动引导（“一键三连”）、创作者自述与推广联系方式会带入无关高频词、干扰主题判断，"
                  "而真正有教学价值的内容（公式推导、建模流程、代码演示、案例解释）多存在于视频画面中。"
                  "因此本周不更换模型，而是从“资源表示是否包含真实教学内容”入手。")

    add_heading(doc, "2. PRISM 原理要点", 2)
    add_body(doc, "论文把图文信息分解为四类：图像独有（如公式布局）、文本独有（如课程名与关键方法词）、"
                  "冗余（图文共同主题）与协同（需两模态结合才能理解）。交互专家层用四个专家分别学习："
                  "独有信息用三元组损失约束，冗余要求遮蔽任一模态后预测仍一致，协同则要求完整预测与单模态扰动预测保持差异；"
                  "自适应融合层结合用户 ID 表示，用 MLP 预测四类权重后加权融合。总损失为 L = Lrec + Lexp，"
                  "可作为插件接入 SASRec、STOSA、InDiRec。论文 Table 3（Yelp、PRISM+InDiRec）显示："
                  "去掉文本独有专家后 R@10 由 0.0422 降至 0.0361，降幅最大，说明文本独有信息贡献突出。")

    add_heading(doc, "3. 环境适配与数据准备", 2)
    add_body(doc, "运行环境为 Windows 11、Python 3.14、PyTorch 2.11+cu126、RTX 4060 Laptop（8 GB）。"
                  "仓库未提供交互文件与两个 512 维特征文件，且默认导入 InDiRec 会触发 faiss 依赖错误，"
                  "导致 STOSA 分支也无法启动，已改为按需导入。随后按 5-core 逻辑重建数据："
                  "22,363 用户、12,101 物品、198,502 条交互（平均序列长 8.88）；用 CLIP ViT-B/32 生成特征，"
                  "文本覆盖 99.9%、图像覆盖 99.8%（缺失物品沿用原脚本的随机向量兜底）。"
                  "外部公开特征虽同为 512 维，但与本地物品仅重合 11/12,101，无法对齐，故未采用。")

    add_heading(doc, "4. 本地对照实验", 2)
    add_body(doc, "固定 STOSA 骨干与训练策略，仅更换图文特征或融合模块，结果见表 1。")
    add_caption(doc, "表 1  本地四类方案测试指标（Beauty / STOSA，单 seed）")
    add_table(doc, ref_tblpr,
              [["变体", "R@10", "R@20", "N@10", "N@20"],
               ["纯 ID STOSA", "0.0724", "0.0985", "0.0421", "0.0486"],
               ["ID + 图文直接相加", "0.0735", "0.1011", "0.0454", "0.0524"],
               ["PRISM + 占位噪声特征", "0.0730", "0.0994", "0.0452", "0.0518"],
               ["PRISM + 真实 CLIP 特征", "0.0750", "0.1032", "0.0463", "0.0534"]],
              [Cm(5.4), Cm(2.4), Cm(2.4), Cm(2.4), Cm(2.4)], accent_rows=(4,))
    add_body(doc, "真实特征相对占位特征四项指标均提升 2.4%～3.8%，最佳轮次由 43 提前至 38；"
                  "与纯 ID 相比，PRISM+真实 CLIP 的 R@10 提升 3.59%、N@10 提升 9.98%。"
                  "但“图文直接相加”已取得大部分增益，PRISM 专家融合在已有图文向量之上的净贡献约 2%。")

    add_heading(doc, "5. 与论文结果的差异", 2)
    add_body(doc, "对照论文 Table 2（Beauty、STOSA），结果见表 2。")
    add_caption(doc, "表 2  论文与本地结果对照")
    add_table(doc, ref_tblpr,
              [["指标", "论文 STOSA", "论文 PRISM+STOSA（相对提升）", "本地 PRISM+CLIP"],
               ["R@10", "0.0648", "0.0762（+17.59%）", "0.0750"],
               ["R@20", "0.0941", "0.1083（+15.09%）", "0.1032"],
               ["N@10", "0.0339", "0.0481（+41.89%）", "0.0463"],
               ["N@20", "0.0385", "0.0549（+42.60%）", "0.0534"]],
              [Cm(2.2), Cm(2.9), Cm(6.2), Cm(3.7)])
    add_body(doc, "本地绝对指标与论文相差 1.6%～4.7%，但相对纯 ID 基线的提升远小于论文。"
                  "原因之一是本地 STOSA 基线已达 0.0724，高于论文的 0.0648（+11.7%），提升空间被压缩。"
                  "为对齐口径，将隐层维度与序列长度由 256/100 调整为 64/50，纯 ID 基线降至 0.0639，"
                  "与论文 0.0648 仅差 1.4%，后续消融均在此配置下进行。此外，本地评测按 trainers.py 的实现"
                  "取 40 个候选计算指标，与论文的框架、候选集与种子数不完全一致，故以“同一实现内部对照”为主要依据。")

    add_heading(doc, "6. 参数修正、消融与多随机种子", 2)
    add_body(doc, "λ 修正后按论文区间调至 (0.05, 0.2, 0.5)，R@10 反而由 0.0673 降至 0.0651；"
                  "“仅保留融合表示”降至 0.0529——代码检查发现 Adaptive_Fusion_Layer 先把序列 item embedding "
                  "求均值成用户级向量再广播，难以替代逐物品图文特征，切断原始通路会丢失细粒度信息；"
                  "训练上限放宽到 150 轮（实际 111 轮早停），R@10 仅由 0.0673 到 0.0674。"
                  "三随机种子（42/1/2）的复核结果见表 3。")
    add_caption(doc, "表 3  三随机种子测试结果（64/50 配置，均值 ± 标准差）")
    add_table(doc, ref_tblpr,
              [["指标", "纯 ID STOSA", "PRISM", "相对变化"],
               ["Recall@10", "0.0630 ± 0.0011", "0.0665 ± 0.0011", "+5.61%（t≈3.92）"],
               ["Recall@20", "0.0911 ± 0.0020", "0.0907 ± 0.0012", "−0.37%"],
               ["NDCG@10", "0.0357 ± 0.0011", "0.0406 ± 0.0005", "+13.62%"],
               ["NDCG@20", "0.0428 ± 0.0013", "0.0467 ± 0.0003", "+9.20%"]],
              [Cm(3.2), Cm(4.2), Cm(4.2), Cm(3.4)])
    add_body(doc, "PRISM 对 Top-10 的召回与排序有稳定正向作用（R@10、N@10、N@20 提升均显著），"
                  "但 Recall@20 基本持平（−0.37%），不宜概括为“所有指标均提升”。")

    add_heading(doc, "7. 数学建模视频场景的应用思路", 2)
    add_body(doc, "把上阶段“简介噪声大”具体化为两个问题：资源表示是否包含真实教学内容，不同模态信息应如何融合。"
                  "可尝试用关键教学画面的 OCR 公式/代码、视觉特征与清洗后的标题简介共同描述视频，"
                  "再参考 PRISM 的独有、冗余、协同分解做加权融合。需注意论文只假定已有物品图文表示，"
                  "不涉及关键帧检测、公式 OCR、片段定位，这部分需自行设计。待验证假设有二："
                  "①补充画面信息能否降低简介噪声、提高标签准确性；②PRISM 式融合相对简单拼接/相加是否仍有足够增益且开销可控。")

    # ---- 二
    add_heading(doc, "二、本周阶段性问题", 1)
    add_body(doc, "1. 本地相对提升与论文仍有约 3 倍差距，评测协议不完全一致。")
    add_body(doc, "2. 公开代码 SASRec/InDiRec 路径仍有依赖或参数问题，当前复现集中于 STOSA。")
    add_body(doc, "3. 模型仍在 Amazon 商品数据上验证，尚未建立数学建模视频的关键帧、公式/代码图像与人工标注集合。")

    # ---- 三
    add_heading(doc, "三、下一步计划", 1)
    add_body(doc, "1. 建设小规模验证样本：清洗标题与简介中的宣传信息，人工核验讲授方法、软件与案例类型，"
                  "并固定测试列表比较“原描述/清洗文本/补充关键画面”三种资源表示。")
    add_body(doc, "2. 研究关键画面提取流程：比较固定间隔采样与画面变化检测，增加教学区域筛选、OCR/公式识别与 CLIP 编码，"
                  "并人工抽样核查有效性。")
    add_body(doc, "3. 验证多模态的真实收益：同一划分下比较文本基线、简单融合与 PRISM 式融合，"
                  "除 R@K、N@K 外记录知识点标注准确性、方法覆盖率、重复资源比例与提取耗时；若收益有限则优先低开销方案。")
    add_body(doc, "4. 完善可重复性：继续排查融合表示插入位置、模态权重训练与不同骨干的实现细节，"
                  "补充随机种子，并统一候选集与指标计算口径。")

    doc.save(DST)
    print("saved", DST)


if __name__ == "__main__":
    main()



