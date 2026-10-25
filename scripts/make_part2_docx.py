"""
Script to generate the Word document:
d:/MSC AIML/Semester 5/SP_PROJECT/4_Progress_Part2_GaitAnalysis_Personalization.docx
Matching the structure, tone, and visual design of 3_Progress_Part1_FallDetection.docx.
"""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    """Set background color of a cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_table_borders_none(table):
    """Remove all table borders."""
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="none"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(tblBorders)

def set_table_borders_grid(table, color="D3D3D3"):
    """Set clean single borders for data tables."""
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:left w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:bottom w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:right w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
            <w:insideV w:val="single" w:sz="4" w:space="0" w:color="{color}"/>
        </w:tblBorders>
    ''')
    tblPr.append(tblBorders)

def add_banner(doc, text_p1, text_p2=None, fill="DCE9F7", p1_size=12.5, p1_bold=True, p1_color=(0x24, 0x40, 0x5e), p2_size=11.5, p2_color=(0x40, 0x55, 0x6f), top_pad=90, bot_pad=90):
    """Add a colored banner table matching 3_Progress_Part1_FallDetection.docx style."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    set_table_borders_none(tbl)
    
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_shading(cell, fill)
    set_cell_margins(cell, top=top_pad, bottom=bot_pad, left=160, right=160)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2 if text_p2 else 0)
    p.paragraph_format.line_spacing = 1.15
    run1 = p.add_run(text_p1)
    run1.bold = p1_bold
    run1.font.name = "Calibri"
    run1.font.size = Pt(p1_size)
    run1.font.color.rgb = RGBColor(*p1_color)
    
    if text_p2:
        p2 = cell.add_paragraph()
        p2.paragraph_format.space_before = Pt(0)
        p2.paragraph_format.space_after = Pt(0)
        p2.paragraph_format.line_spacing = 1.15
        run2 = p2.add_run(text_p2)
        run2.bold = False
        run2.font.name = "Calibri"
        run2.font.size = Pt(p2_size)
        run2.font.color.rgb = RGBColor(*p2_color)

def add_body_p(doc, text="", bold_prefix=None, space_after=6):
    """Add standard body paragraph with 10.5pt Calibri."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10.5)
    
    if text:
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
    return p

def main():
    doc = docx.Document()
    
    # Page setup: 1 inch left/right, 0.7 inch top/bottom
    for s in doc.sections:
        s.top_margin = Inches(0.7)
        s.bottom_margin = Inches(0.7)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # -------------------------------------------------------------
    # 1. Main Title Banner (fill=CBDDF2)
    # -------------------------------------------------------------
    add_banner(
        doc,
        text_p1="Smart Rollator — Progress So Far",
        text_p2="Part 2: Gait Analysis, Personalization & Longitudinal Decline Tracking  ·  plain-language summary",
        fill="CBDDF2",
        p1_size=19.0,
        p1_bold=True,
        p1_color=(0x24, 0x40, 0x5e),
        p2_size=11.5,
        p2_color=(0x40, 0x55, 0x6f),
        top_pad=200,
        bot_pad=200
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 2. Section: What this part is (fill=DCE9F7)
    # -------------------------------------------------------------
    add_banner(doc, "What this part is", fill="DCE9F7")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    add_body_p(
        doc,
        "The whole project puts a small “brain” on a walker that can tell when an elderly person is moving in an unsafe way. "
        "While Part 1 built the immediate emergency brain (“has a fall already happened?”), this second part builds the early-warning brain — "
        "the piece that continuously observes how a person walks, catches unstable movement before a fall occurs, and tracks whether their mobility is gradually deteriorating over weeks. "
        "No elderly person falls without prior subtle signs: walking slows down, steps become irregular, side-to-side sway widens, and cadence fluctuates. "
        "This part was built and tested entirely on a laptop using motion-sensor data across 47 real people (young adults and real elderly citizens), adhering strictly to the team’s shared contract: 2.5-second clips (125 readings at 50 readings per second)."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 3. Section: What we built (fill=DCEFE3)
    # -------------------------------------------------------------
    add_banner(doc, "What we built", fill="DCEFE3")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    add_body_p(
        doc,
        "Rather than passing raw, uninterpretable sensor spikes to a black-box model, we extract 26 clinically recognized biomechanical features from each 2.5-second motion clip. These measure four pillars of walking health: "
        "(1) Cadence & Rhythm (stepping pace and frequency purity); "
        "(2) Step & Stride Regularity (mathematical autocorrelation measuring how identical consecutive steps and full strides are); "
        "(3) Variability & Impact (how evenly the foot strikes the ground and step-to-step timing variation); and "
        "(4) Balance & Lateral Sway (side-to-side wobble RMS, rotational gyro sway, and jerkiness of movement)."
    )
    
    add_body_p(doc, "a machine-learning classifier (calibrated Random Forest) trained to instantly distinguish steady walking from trips, slips, stumbles, and sudden balance losses.", bold_prefix="Instability & Stumble Detector — ")
    add_body_p(doc, "an individualized anomaly-detection engine (Isolation Forest + statistical z-scores) fitted for each person. Because an 80-year-old naturally walks slower and with a different rhythm than a 22-year-old, fixed universal thresholds fail. The system learns each user’s unique baseline during normal walking, and flags deviations against their own personal normal.", bold_prefix="Per-User Personalization — ")
    add_body_p(doc, "a composite 0–100 score reflecting overall gait quality. A 7-day rolling linear regression detects downward trends (stable, declining, improving) and automatically alerts caregivers before an emergency happens.", bold_prefix="Longitudinal Decline Tracker & Mobility Health Score — ")
    add_body_p(doc, "a clean Python interface exposing predict_gait() and get_mobility_report() for Member 3’s companion mobile application and caregiver dashboard.", bold_prefix="Public Deployment API — ")
    
    add_body_p(
        doc,
        "Combined, our harmonized gait dataset contains 17,902 clips across 47 subjects (38 from SisFall including 15 real elderly individuals, and 9 from MobiFall). "
        "Of these, 17,165 clips represent normal, safe walking and 737 clips capture verified stumbles, trips, and balance perturbations."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 4. Section: Is the data trustworthy? (quality checks) (fill=F7E5D6)
    # -------------------------------------------------------------
    add_banner(doc, "Is the data trustworthy? (quality checks)", fill="F7E5D6")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    add_body_p(doc, "Before trusting any clinical gait model, the pipeline runs automatic quality checks — all passed:")
    add_body_p(doc, "When a person stands still or pauses between trials, the sensor must read Earth’s gravity (1.00g). Across all 47 participants, our harmonized median still reading measured exactly 0.997g — confirming units and physical scaling are correct.", bold_prefix="Physical Gravity Scaling — ")
    add_body_p(doc, "Every recording sits on a perfectly even 50 readings per second timeline (125 readings per 2.5-second window, 50% overlap), matching Member 1’s fall CNN contract with zero gaps.", bold_prefix="Sampling Rate Alignment — ")
    add_body_p(doc, "Phone accelerometers max out at ±2.0g on hard movements (the key finding from Part 1). We deliberately clip acceleration at ±2.0g during feature extraction, ensuring algorithms developed here transfer seamlessly to inexpensive smartphone and rollator hardware.", bold_prefix="Phone Sensor Saturation Matching — ")
    add_body_p(doc, "The model is evaluated using 5-Fold GroupKFold strictly partitioned by participant ID. The brain is always tested on people it never learned from (zero subject overlap between training and testing folds).", bold_prefix="Strict Subject-Independent Validation — ")
    add_body_p(doc, "5 out of 5 automated unit and regression tests pass in ~1 second, verifying contract bounds, numerical stability, and that single-window inference latency is ~108 ms (well inside our 1.24-second streaming budget).", bold_prefix="Automated Test Suite — ")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 5. Section: The results (fill=E7DEF6)
    # -------------------------------------------------------------
    add_banner(doc, "The results", fill="E7DEF6")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    add_body_p(doc, "Three simple ideas explain the numbers below:")
    add_body_p(doc, "of all real stumbles and unstable events, how many the brain catches. Higher is better.", bold_prefix="Instability caught — ")
    add_body_p(doc, "of all normal, safe walking, how much it correctly ignores. Higher means fewer false alarms.", bold_prefix="Normal left alone — ")
    add_body_p(doc, "how often it cries wolf on steady walking. Lower is better.", bold_prefix="False alarms — ")
    
    # Results Table 1 (Instability Classifier)
    t1 = doc.add_table(rows=2, cols=4)
    t1.autofit = False
    set_table_borders_grid(t1)
    
    headers = ["Test (all fair: unseen people)", "Instability caught", "Normal left alone", "False alarms"]
    widths = [Inches(3.0), Inches(1.2), Inches(1.2), Inches(1.1)]
    
    for c_i, h in enumerate(headers):
        cell = t1.rows[0].cells[c_i]
        cell.width = widths[c_i]
        set_cell_shading(cell, "DCE9F7")
        set_cell_margins(cell, top=60, bottom=60, left=110, right=110)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(10.0)
        
    row_data = ["5-Fold GroupKFold (47 unseen subjects)", "88.5%", "99.7%", "0.3%"]
    for c_i, val in enumerate(row_data):
        cell = t1.rows[1].cells[c_i]
        cell.width = widths[c_i]
        set_cell_margins(cell, top=60, bottom=60, left=110, right=110)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(val)
        run.font.name = "Calibri"
        run.font.size = Pt(10.0)
        if c_i > 0:
            run.bold = True
            
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    add_body_p(
        doc,
        "Overall Classifier Metrics: ROC-AUC = 0.9891 (98.9%), Precision = 92.0%, F1 Score = 0.9018, and Overall Accuracy = 99.2%. "
        "The top features separating steady walking from stumbles were step impact intensity variation (15.8%), intensity CV (13.3%), gyro sway RMS (7.6%), and acceleration jerk (5.9%)."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    # Results Table 2 (Personalization Breakdown)
    add_body_p(doc, "Breakdown across demographic cohorts:", bold_prefix="Personalized Anomaly Detection — ")
    t2 = doc.add_table(rows=3, cols=4)
    t2.autofit = False
    set_table_borders_grid(t2)
    
    headers_t2 = ["Demographic Cohort", "Subjects", "Normal Walking Clips", "Stumble Anomalies Caught"]
    widths_t2 = [Inches(2.5), Inches(1.1), Inches(1.5), Inches(1.4)]
    
    for c_i, h in enumerate(headers_t2):
        cell = t2.rows[0].cells[c_i]
        cell.width = widths_t2[c_i]
        set_cell_shading(cell, "DCE9F7")
        set_cell_margins(cell, top=60, bottom=60, left=110, right=110)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(10.0)
        
    rows_t2 = [
        ["Young Adults (SA01–SA23)", "23 people", "8,869 clips", "92.1% (504 / 547)"],
        ["Real Elderly (SE01–SE15)", "15 people", "5,504 clips", "90.5% (172 / 190)"]
    ]
    for r_i, r_data in enumerate(rows_t2):
        for c_i, val in enumerate(r_data):
            cell = t2.rows[r_i + 1].cells[c_i]
            cell.width = widths_t2[c_i]
            set_cell_margins(cell, top=60, bottom=60, left=110, right=110)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = "Calibri"
            run.font.size = Pt(10.0)
            if c_i == 3:
                run.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    add_body_p(
        doc,
        "Across all 737 stumble trials, per-user calibrated baselines successfully caught 676 stumbles as personal anomalies (91.7% detection rate) — "
        "without the baseline model ever having seen that user stumble before."
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 6. Callout Banner: KEY FINDING (fill=FFF4CC)
    # -------------------------------------------------------------
    add_banner(
        doc,
        text_p1="KEY FINDING  Fixed universal thresholds fail on senior citizens, but relative biomechanical change is unmistakable. During unstable walking and stumbles, step regularity drops by 41.5% (from 0.390 to 0.228), stride variability jumps by 44.1% (from 10.2% to 14.7%), and impact jerk rises by 41.0% (from 3.88 to 5.47). By comparing senior citizens against their own calibrated baseline rather than a young adult reference, the system eliminates false alarms while catching 91.7% of stumble hazards.",
        fill="FFF4CC",
        p1_size=10.5,
        p1_bold=False,
        p1_color=(0x00, 0x00, 0x00),
        top_pad=120,
        bot_pad=120
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 7. Section: Longitudinal Decline Tracking (SE01 Case Study)
    # -------------------------------------------------------------
    add_banner(doc, "Simulated Longitudinal Mobility Drift (4-Week Tracking)", fill="DCE9F7")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    add_body_p(
        doc,
        "To evaluate whether the system can spot gradual frailty over weeks, we simulated a 28-day clinical drift on elderly subject SE01 "
        "(gradually introducing step shortening, widening lateral sway, and rising timing variability based on published geriatric mobility literature):"
    )
    
    t3 = doc.add_table(rows=5, cols=5)
    t3.autofit = False
    set_table_borders_grid(t3)
    
    headers_t3 = ["Timeline", "Mobility Score", "Step Regularity", "Stride CV %", "System Status"]
    widths_t3 = [Inches(1.5), Inches(1.2), Inches(1.2), Inches(1.1), Inches(1.5)]
    
    for c_i, h in enumerate(headers_t3):
        cell = t3.rows[0].cells[c_i]
        cell.width = widths_t3[c_i]
        set_cell_shading(cell, "DCE9F7")
        set_cell_margins(cell, top=60, bottom=60, left=110, right=110)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(10.0)
        
    rows_t3 = [
        ["Week 1 (Baseline)", "94.1 / 100", "0.484", "16.58%", "Normal (Green)"],
        ["Week 2 (Mild change)", "91.3 / 100", "0.448", "17.72%", "Normal (Green)"],
        ["Week 3 (Emerging drift)", "81.6 / 100", "0.380", "20.35%", "Monitoring (Amber)"],
        ["Week 4 (Frailty drift)", "69.9 / 100", "0.295", "24.12%", "CAREGIVER ALERT (Red)"]
    ]
    for r_i, r_data in enumerate(rows_t3):
        for c_i, val in enumerate(r_data):
            cell = t3.rows[r_i + 1].cells[c_i]
            cell.width = widths_t3[c_i]
            set_cell_margins(cell, top=60, bottom=60, left=110, right=110)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = "Calibri"
            run.font.size = Pt(10.0)
            if c_i == 4 and "ALERT" in val:
                run.bold = True
                run.font.color.rgb = RGBColor(0x99, 0x00, 0x00)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    add_body_p(
        doc,
        "The decline tracker detected a downward slope of -9.93 points/week (far steeper than our safety threshold of -3.0 points/week), "
        "automatically firing a caregiver notification: “Warning: Progressive mobility decline detected. Caregiver check recommended.” "
        "The 4-panel longitudinal trend plot below illustrates this multi-week drift:"
    )
    
    chart_path = Path("d:/MSC AIML/Semester 5/SP_PROJECT/m2-gait-features/data/reports/mobility_decline_trend.png")
    if chart_path.exists():
        pic_p = doc.add_paragraph()
        pic_p.paragraph_format.space_before = Pt(4)
        pic_p.paragraph_format.space_after = Pt(6)
        pic_p.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
        pic_p.add_run().add_picture(str(chart_path), width=Inches(6.0))
        
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 8. Section: Honest limits (state these openly) (fill=DCE9F7)
    # -------------------------------------------------------------
    add_banner(doc, "Honest limits (state these openly)", fill="DCE9F7")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    add_body_p(doc, "Public datasets cover falls and daily activities well, but there is no public dataset of continuous gait recorded directly from a rolling walker frame. The models currently train on waist and pocket IMUs.", bold_prefix="Public Datasets Lack Rollator Mounting — ")
    add_body_p(doc, "Multi-week decline in the elderly occurs over months or years. Our 28-day drift was simulated on top of real elderly baselines (based on clinical literature), and is clearly documented as a proof-of-concept simulation.", bold_prefix="Longitudinal Drift is Simulated — ")
    add_body_p(doc, "The models run fast (108 ms warm latency on CPU) and unit tests pass, but empirical validation on a physical smartphone taped to an actual rolling walker is the final step required before live deployment.", bold_prefix="Final Walker Validation Needed — ")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    
    # -------------------------------------------------------------
    # 9. Section: What's next (Part 3 & Integration) (fill=DCEFE3)
    # -------------------------------------------------------------
    add_banner(doc, "What’s next (Part 3 & Integration)", fill="DCEFE3")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    
    add_body_p(
        doc,
        "Part 2 completes the gait analysis, personalization, and decline tracking models. What remains is connecting these pieces into the live product: "
        "(1) Handoff predict_gait() and get_mobility_report() to Member 3 for companion app integration and caregiver dashboard display; "
        "(2) Combine Member 1’s fall detector with Member 2’s gait scores in the backend decision engine with the 30-second “I’m OK” cancel window; and "
        "(3) Record our own 15-minute custom dataset with a smartphone taped to an actual rollator frame (normal walk, shuffling, limping, and stumbles) to validate walker-mounted performance."
    )
    
    output_path = Path("d:/MSC AIML/Semester 5/SP_PROJECT/4_Progress_Part2_GaitAnalysis_Personalization.docx")
    doc.save(str(output_path))
    print(f"Document successfully saved to {output_path}")

if __name__ == "__main__":
    main()
