"""Explicit empirical configuration and runtime checkpoint regressions."""
import json
import pandas as pd
import pytest
from algo.diet_structure_priors import get_common_uncommon_sets
from dmp4p_runtime.checkpoint import SlotCheckpoint

def test_frequency_configuration_is_not_replaced_by_default(tmp_path):
    a=tmp_path/'a.csv'; b=tmp_path/'b.csv'
    pd.DataFrame({'food_id':[11,22],'pct_days':[8.,0.2]}).to_csv(a,index=False)
    pd.DataFrame({'food_id':[11,22],'pct_days':[0.2,8.]}).to_csv(b,index=False)
    def cfg(p):return {'paths':{'food_frequency':str(p)},'pool':{'common_pct_days_min':5,'uncommon_pct_days_max':1}}
    assert get_common_uncommon_sets(cfg(a))==({11},{22})
    assert get_common_uncommon_sets(cfg(b))==({22},{11})
    assert get_common_uncommon_sets(cfg(a))==({11},{22})

def test_runtime_checkpoint_preserves_identity_and_consumed_budget(tmp_path):
    cp=SlotCheckpoint(tmp_path/'cp.json',{'experiment':'new_prior','run':'one'})
    cp.save({'phase':'boundary','consumed_s':31.2,'task_wall_s':210,'days':[{'plan_uid':'day1'}],'banned_food_ids':[11]})
    d=cp.load()
    assert d['consumed_s']==31.2 and d['days'][0]['plan_uid']=='day1' and d['banned_food_ids']==[11]
    with pytest.raises(ValueError):
        SlotCheckpoint(cp.path,{'experiment':'old_prior','run':'one'}).load()

def test_runtime_inflight_crash_cannot_extend_budget(tmp_path):
    cp=SlotCheckpoint(tmp_path/'cp.json',{'experiment':'new','run':'one'})
    cp.save({'phase':'inflight','consumed_s':31.2,'task_wall_s':210,'days':[{'plan_uid':'day1'}],'banned_food_ids':[11]})
    assert cp.load()['consumed_s']==210
