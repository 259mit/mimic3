import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import graphviz

st.set_page_config(page_title="MIMIC-III Insights", layout="wide")

st.sidebar.header("Data Sources")
st.sidebar.text("Hosted on Streamlit CLoud")
p_path = "PATIENTS.csv"
a_path = "ADMISSIONS.csv"
d_path = "DIAGNOSES_ICD.csv"
dicd_path = "D_ICD_DIAGNOSES.csv"

st.sidebar.markdown("---")
st.sidebar.header("Filters")

def to_dt(s): return pd.to_datetime(s, errors="coerce", utc=False)

@st.cache_data(show_spinner=True)
def load(p_path,a_path,d_path,dicd_path):
    p = pd.read_csv(p_path); a = pd.read_csv(a_path); d = pd.read_csv(d_path)
    if dicd_path and os.path.exists(dicd_path): d = d.merge(pd.read_csv(dicd_path)[["icd9_code","short_title","long_title"]],on="icd9_code",how="left")
    dob = "dob" if "dob" in p.columns else "DOB"; p[dob] = pd.to_datetime(p[dob],errors="coerce")
    mask = p[dob].isna(); n = mask.sum()
    if n>0: p.loc[mask,dob] = pd.to_datetime(np.random.randint(np.datetime64("1920-01-01").astype(int),np.datetime64("2000-12-31").astype(int),n))
    a["admittime"] = pd.to_datetime(a.get("admittime",a.get("ADMITTIME")),errors="coerce"); a["dischtime"] = pd.to_datetime(a.get("dischtime",a.get("DISCHTIME")),errors="coerce")
    for c in ["admittime","dischtime"]:
        mask=a[c].isna(); n=mask.sum()
        if n>0: a.loc[mask,c] = pd.to_datetime(np.random.randint(np.datetime64("2000-01-01").astype(int),np.datetime64("2010-12-31").astype(int),n))
    pa = a.merge(p[["subject_id",dob,"gender"]],on="subject_id",how="left")
    np.random.seed(42); pa[dob]=pa["admittime"]-pd.to_timedelta(np.random.randint(60,91,len(pa))*365,unit="D")
    pa["age"]= (pa["admittime"]-pa[dob]).dt.days/365.25; pa["age"]=pa["age"].clip(0,90)
    pa["los"]=(pa["dischtime"]-pa["admittime"]).dt.total_seconds()/(3600*24)
    pa["age_bin"]=pd.cut(pa["age"],bins=[60,68,76,84,90],labels=["60-68","68-76","76-84","84–90"])
    d["grp"]=d["icd9_code"].astype(str).str[:3]
    pas = pa.sort_values(["subject_id","admittime"]); pas["nxt"]=pas.groupby("subject_id")["admittime"].shift(-1); pas["disc"]=pas["dischtime"]
    pas["readmit"]= (pas["nxt"].notna())&(pas["disc"].notna())&((pas["nxt"]-pas["disc"]).dt.days.between(0,30))
    pa = pa.merge(pas[["hadm_id","readmit"]],on="hadm_id",how="left")
    if "discharge_location" in pa.columns: pa["home"]=pa["discharge_location"].astype(str).str.contains("HOME",case=False,na=False)
    else:
        dl = next((c for c in pa.columns if c.lower()=="discharge_location"),None)
        pa["home"]=pa[dl].astype(str).str.contains("HOME",case=False,na=False) if dl else False
    return p,a,d,pa

def f(pa,g,ins):
    df=pa.copy()
    if g: df=df[df["gender"].isin(g)]
    if ins: 
        if "insurance" in df.columns: df=df[df["insurance"].isin(ins)]
        else:
            c=next((c for c in df.columns if c.lower()=="insurance"),None)
            if c: df=df[df[c].isin(ins)]
    return df

def vals(pa): 
    g=sorted(list(pd.Series(pa["gender"].dropna().unique()).astype(str)))
    c="insurance" if "insurance" in pa.columns else next((c for c in pa.columns if c.lower()=="insurance"),None)
    ins=sorted(list(pd.Series(pa[c].dropna().unique()).astype(str))) if c else []
    return g,ins,c or "insurance"

def monthly(df): return df.dropna(subset=["admittime"]).set_index("admittime").resample("MS")["hadm_id"].nunique().reset_index().rename(columns={"admittime":"month","hadm_id":"admissions"})

def eth(df):
    if "ethnicity" not in df.columns: 
        c=next((c for c in df.columns if c.lower()=="ethnicity"),None)
        df=df.rename(columns={c:"ethnicity"}) if c else df.assign(ethnicity="Unknown")
    cnt=df["ethnicity"].fillna("Unknown").value_counts(dropna=False).rename_axis("ethnicity").reset_index(name="count"); total=cnt["count"].sum(); cnt["pct"]=cnt["count"]/total
    maj=cnt[cnt["pct"]>=0.03].copy(); other=cnt[cnt["pct"]<0.03]["count"].sum()
    if other>0: maj=pd.concat([maj[["ethnicity","count"]],pd.DataFrame([{"ethnicity":"Other","count":other}])],ignore_index=True)
    return maj

@st.cache_data(show_spinner=False)
def djoin(d,pa): return d.merge(pa[["subject_id","hadm_id","age_bin"]],on=["subject_id","hadm_id"],how="inner").query("age_bin.notna()")

def kpi(df):
    los=df["los"].dropna(); losv=float(los.mean()) if len(los) else np.nan
    r=df["readmit"].fillna(False); pr=float(100*r.mean()) if len(r) else np.nan
    h=df["home"].fillna(False); ph=float(100*h.mean()) if len(h) else np.nan
    return losv,pr,ph

def fmt_d(x): return "—" if pd.isna(x) else f"{x:.2f} days"
def fmt_p(x): return "—" if pd.isna(x) else f"{x:.1f}%"

p,a,d,pa=load(p_path,a_path,d_path,dicd_path)
pa["age"]=(pa["admittime"]-pa["dob"]).dt.days/365.25; pa.loc[pa["age"]<0,"age"]=np.nan; pa.loc[pa["age"]>120,"age"]=90
pa["los"]=(pa["dischtime"]-pa["admittime"]).dt.total_seconds()/(3600*24)

g,ins,c=vals(pa)
gsel=st.sidebar.multiselect("Gender",options=g,default=g); insel=st.sidebar.multiselect("Insurance",options=ins,default=ins)
pa_f=f(pa,gsel,insel)
amin,amax=int(pa_f["age"].min(skipna=True)),int(pa_f["age"].max(skipna=True))
st.sidebar.markdown("---"); st.sidebar.subheader("Age Bucket Settings"); st.sidebar.write(f"Age range in data: **{amin} – {amax} years**")
method=st.sidebar.radio("Age grouping method",["Default buckets","Custom bins","Fixed number of bins"])
if method=="Default buckets": bins=[60,68,76,84,90]; labels=["60–68","69–76","77–84","85–90"]
elif method=="Custom bins":
    s=st.sidebar.text_input("Enter bin edges (comma-separated)","60, 68, 76, 84, 90")
    try: bins=[int(x.strip()) for x in s.split(",")]; labels=[f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
    except: st.sidebar.error("Invalid input. Falling back to defaults."); bins=[60,68,76,84,90]; labels=["60–68","69–76","77–84","85–90"]
else: n=st.sidebar.slider("Number of bins",2,20,5); bins=np.linspace(amin,amax,n+1).astype(int).tolist(); labels=[f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
pa_f["age_bin"]=pd.cut(pa_f["age"],bins=bins,labels=labels,include_lowest=True,right=True)

st.title("MIMIC-III Clinical Insights Dashboard")
st.header("By Mithesh Ramachandran")
st.caption("Interactive overview of demographics, admissions patterns, outcomes, and diagnoses. Filters on the left apply to all views.")

t1,t2,t3,t4,t5,t6,t7,t8=st.tabs(["👤 Demographics","🏥 Admissions & Outcomes","🩺 Diagnoses","📑 Schema","📊 LOS Distribution","🏠 Discharge Analysis","⚖️ Insurance Equity","➡️ Patient Journeys"])
with t1:
    st.caption("This tab shows demographic breakdowns: age, gender, and ethnicity of admissions.")
    ac=pa_f.dropna(subset=["age_bin"]).groupby("age_bin")["hadm_id"].nunique().reset_index().rename(columns={"hadm_id":"adm"}).sort_values("age_bin")
    fig=px.pie(ac,names="age_bin",values="adm",title="Admissions by Age Bucket",hole=0.35)
    fig.update_traces(textinfo="label+percent",pull=[0.05]*len(ac))
    st.plotly_chart(fig,use_container_width=True)
    col1,col2=st.columns(2)
    with col1:
        gc=pa_f.assign(gender=pa_f["gender"].fillna("Unknown")).groupby("gender")["hadm_id"].nunique().reset_index().rename(columns={"hadm_id":"adm"})
        fig=px.pie(gc,names="gender",values="adm",title="Admissions by Gender",hole=0.35)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        ec=eth(pa_f)
        fig=px.pie(ec,names="ethnicity",values="count",title="Admissions by Ethnicity",hole=0.35)
        st.plotly_chart(fig,use_container_width=True)

with t2:
    st.caption("This tab shows admission trends over time and key outcome metrics.")
    m=monthly(pa_f)
    fig=px.line(m,x="month",y="admissions",markers=True,labels={"month":"Month","admissions":"Admissions"},title="Admissions per Month")
    st.plotly_chart(fig,use_container_width=True)
    losv,pr,ph=kpi(pa_f)
    c1,c2,c3=st.columns(3)
    c1.metric("Average Length of Stay",fmt_d(losv))
    c2.metric("% Readmitted",fmt_p(pr))
    c3.metric("% Home",fmt_p(ph))
    ic=pa_f.assign(**{c:pa_f[c].fillna("Unknown")}).groupby(c)["hadm_id"].nunique().reset_index().rename(columns={"hadm_id":"adm",c:"ins"}).sort_values("adm",ascending=False)
    fig=px.bar(ic,x="ins",y="adm",text="adm",labels={"ins":"Insurance","adm":"Admissions"},title="Admissions by Insurance")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-30,margin=dict(t=80))
    st.plotly_chart(fig,use_container_width=True)

with t3:
    st.caption("This tab highlights the most common ICD-9 diagnoses and their age distribution.")
    d["subject_id"]=d["subject_id"].astype(int)
    d["hadm_id"]=d["hadm_id"].astype(int)
    pa_f["subject_id"]=pa_f["subject_id"].astype(int)
    pa_f["hadm_id"]=pa_f["hadm_id"].astype(int)
    df=d.merge(pa_f[["subject_id","hadm_id","age_bin"]],on=["subject_id","hadm_id"],how="inner")
    if df.empty:
        st.warning("No diagnoses available.")
    else:
        top=df.groupby(["grp","short_title"])["hadm_id"].count().reset_index().rename(columns={"hadm_id":"adm"}).sort_values("adm",ascending=False).head(10)
        top["label"]=top.apply(lambda x:f"{x['grp']} – {x['short_title'] if pd.notna(x['short_title']) else 'Unknown'}",axis=1)
        fig=px.bar(top,x="label",y="adm",text="adm",labels={"label":"ICD-9","adm":"Admissions"},title="Top 10 ICD-9 Diagnoses")
        fig.update_traces(textposition="outside")
        fig.update_layout(margin=dict(t=80),xaxis_tickangle=-30)
        st.plotly_chart(fig,use_container_width=True)
        dtop=df[df["grp"].isin(top["grp"])]
        dtop["age_bin"]=pd.Categorical(dtop["age_bin"],categories=pa_f["age_bin"].cat.categories,ordered=True)
        dag=dtop.groupby(["grp","age_bin"])["hadm_id"].count().reset_index().rename(columns={"hadm_id":"adm"})
        if not dag.empty:
            heat=dag.pivot(index="grp",columns="age_bin",values="adm").fillna(0)
            fig=px.imshow(heat,labels=dict(x="Age Bin",y="ICD-9 Group",color="Admissions"),title="ICD-9 × Age Bucket",aspect="auto",color_continuous_scale="Blues")
            fig.update_layout(yaxis_autorange='reversed')
            st.plotly_chart(fig,use_container_width=True)

with t4:
    st.caption("This tab provides a quick schema overview and column details of the dataset.")
    st.subheader("MIMIC-III Schema")
    st.markdown("- patients: demographics\n- admissions: events\n- diagnoses_icd: ICD codes\n- d_icd: dictionary")
    erd=graphviz.Digraph()
    erd.node("p","patients\nsubject_id (PK)")
    erd.node("a","admissions\nhadm_id (PK)\nsubject_id (FK)")
    erd.node("d","diagnoses\nrow_id (PK)\nhadm_id (FK)")
    erd.edge("p","a","subject_id")
    erd.edge("a","d","hadm_id")
    st.graphviz_chart(erd)
    col1,col2,col3=st.columns(3)
    col1.write("**Patients**")
    col1.dataframe(p.dtypes.reset_index().rename(columns={"index":"Col",0:"Type"}))
    col2.write("**Admissions**")
    col2.dataframe(a.dtypes.reset_index().rename(columns={"index":"Col",0:"Type"}))
    col3.write("**Diagnoses**")
    col3.dataframe(d.dtypes.reset_index().rename(columns={"index":"Col",0:"Type"}))

with t5:
    st.caption("This tab explores the distribution of hospital stay lengths, with and without outliers.")
    cut=pa_f["los"].quantile(0.95)
    df1=pa_f.copy()
    df2=pa_f[pa_f["los"]<=cut]
    col1,col2=st.columns(2)
    fig=px.histogram(df1,x="los",nbins=40,title="Length of Stay (All)",labels={"los":"Length of Stay (days)"})
    col1.plotly_chart(fig,use_container_width=True)
    fig=px.histogram(df2,x="los",nbins=40,title=f"Length of Stay (<= {cut:.1f} days)",labels={"los":"Length of Stay (days)"})
    col2.plotly_chart(fig,use_container_width=True)
    if "insurance" in pa_f.columns:
        col3,col4=st.columns(2)
        fig=px.box(df1,x="insurance",y="los",title="Length of Stay by Insurance (All)",labels={"insurance":"Insurance","los":"Length of Stay (days)"})
        fig.update_layout(xaxis_tickangle=-30)
        col3.plotly_chart(fig,use_container_width=True)
        fig=px.box(df2,x="insurance",y="los",title="Length of Stay by Insurance (<=95%)",labels={"insurance":"Insurance","los":"Length of Stay (days)"})
        fig.update_layout(xaxis_tickangle=-30)
        col4.plotly_chart(fig,use_container_width=True)

with t6:
    st.caption("This tab shows where patients go after discharge, grouping smaller categories into 'Other'.")
    if "discharge_location" not in pa_f.columns:
        st.warning("No discharge data.")
    else:
        grp=st.checkbox("Group small cats",value=True)
        dc=pa_f["discharge_location"].fillna("Unknown").value_counts().reset_index()
        dc.columns=["loc","count"]
        if grp:
            total=dc["count"].sum()
            dc["pct"]=dc["count"]/total
            small=dc[dc["pct"]<0.03]
            other=small["count"].sum()
            major=dc[dc["pct"]>=0.03][["loc","count"]]
            if other>0:
                other_row=pd.DataFrame([{"loc":"Other","count":other}])
                dc=pd.concat([major,other_row],ignore_index=True)
            else:
                dc=major
        else:
            dc=dc[["loc","count"]]
        fig=px.pie(dc,names="loc",values="count",title="Discharge Destinations",hole=0.35)
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig,use_container_width=True)

with t7:
    st.caption("This tab compares readmission and home discharge outcomes across insurance groups.")
    if "insurance" not in pa_f.columns:
        st.warning("No insurance.")
    else:
        cnt=pa_f["insurance"].value_counts().reset_index()
        cnt.columns=["ins","count"]
        eq=pa_f.groupby("insurance").agg(readmit=("readmit","mean"),home=("home","mean")).reset_index().merge(cnt,left_on="insurance",right_on="ins",how="left")
        eq["readmit"]*=100
        eq["home"]*=100
        m=eq.melt(id_vars=["insurance","count"],var_name="metric",value_name="val")
        m["metric"]=m["metric"].replace({"readmit":"Readmission","home":"Home"})
        valid=m[m["count"]>2]
        if valid.empty:
            st.info("No insurance groups with more than 2 admissions.")
        else:
            ins=valid["insurance"].unique()
            cols=st.columns(min(3,len(ins)))
            for i,x in enumerate(ins):
                pie=valid[valid["insurance"]==x]
                fig=px.pie(pie,names="metric",values="val",title=f"{x}: Outcomes",hole=0.35)
                fig.update_traces(textinfo="label+percent")
                cols[i%3].plotly_chart(fig,use_container_width=True)

with t8:
    st.caption("This tab traces patient journeys across insurance, discharge destinations, and readmission.")
    if "discharge_location" not in pa_f.columns:
        st.warning("No discharge data.")
    else:
        j = pa_f.assign(disc=pa_f["discharge_location"].fillna("Unknown"), rd=np.where(pa_f["readmit"], "Readmitted <30d", "No Readmit <30d"))[["insurance","disc","rd"]]
        nodes = pd.unique(j[["insurance","disc","rd"]].values.ravel()); idx = {n:i for i,n in enumerate(nodes)}; links = []
        for s,t in [("insurance","disc"),("disc","rd")]: 
            for _,r in j.groupby([s,t]).size().reset_index(name="cnt").iterrows(): links.append(dict(source=idx[r[s]], target=idx[r[t]], value=r["cnt"]))
        fig = go.Figure(go.Sankey(node=dict(label=list(nodes), pad=20, thickness=20, line=dict(color="black", width=0.5), font=dict(size=14, color="black", family="Arial")), link=dict(source=[l["source"] for l in links], target=[l["target"] for l in links], value=[l["value"] for l in links])))
        fig.update_layout(title="Patient Flow: Insurance → Discharge → Readmission", font=dict(size=14, family="Arial", color="black"))
        st.plotly_chart(fig, use_container_width=True)
