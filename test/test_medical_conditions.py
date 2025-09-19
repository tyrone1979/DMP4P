import unittest
from utils.medical_condition_cal import *


class MyTestCase(unittest.TestCase):
    def __init__(self, methodName="Medical Conditions Test"):
        super().__init__(methodName)
        self.judge = MedicalConditions()

    def test_weight(self):
        BMI, WAIST, gender = 25, 102, 1
        result = self.judge.judge_weight(BMI, WAIST, gender)
        self.assertEqual(1, result['low_calorie'])
        self.assertEqual(1, result['over_weight'])
        self.assertEqual(0, result['under_weight'])
        self.assertEqual(0, result['high_calorie'])
        BMI, WAIST, gender = 25, 88, 2
        result = self.judge.judge_weight(BMI, WAIST, gender)
        self.assertEqual(1, result['low_calorie'])
        self.assertEqual(1, result['over_weight'])
        self.assertEqual(0, result['under_weight'])
        self.assertEqual(0, result['high_calorie'])
        BMI = 18
        result = self.judge.judge_weight(BMI, WAIST, gender)
        self.assertEqual(0, result['low_calorie'])
        self.assertEqual(0, result['over_weight'])
        self.assertEqual(1, result['high_calorie'])
        self.assertEqual(1, result['under_weight'])

    def test_judge_hypertension(self):
        systolic, diastolic = 130, 70
        result = self.judge.judge_hypertension(systolic, diastolic)
        self.assertEqual(1, result['low_sodium'])
        self.assertEqual(1, result['high_potassium'])
        self.assertEqual(1, result['blood_pressure'])
        systolic, diastolic = 120, 80
        result = self.judge.judge_hypertension(systolic, diastolic)
        self.assertEqual(1, result['low_sodium'])
        self.assertEqual(1, result['high_potassium'])
        self.assertEqual(1, result['blood_pressure'])
        systolic, diastolic = 120, 70
        result = self.judge.judge_hypertension(systolic, diastolic)
        self.assertEqual(0, result['low_sodium'])
        self.assertEqual(0, result['high_potassium'])
        self.assertEqual(0, result['blood_pressure'])

    def test_judge_high_low_density_lipoprotein(self):
        ldl = 3.4
        result = self.judge.judge_high_low_density_lipoprotein(ldl)
        self.assertEqual(1, result['low_cholesterol'])
        self.assertEqual(1, result['high_fiber'])
        self.assertEqual(1, result['low_saturated_fat'])
        self.assertEqual(1, result['low_density_lipoprotein'])
        ldl = 3.3
        result = self.judge.judge_high_low_density_lipoprotein(ldl)
        self.assertEqual(0, result['low_cholesterol'])
        self.assertEqual(0, result['high_fiber'])
        self.assertEqual(0, result['low_saturated_fat'])
        self.assertEqual(0, result['low_density_lipoprotein'])

    def test_judge_high_blood_urea_nitrogen(self):
        blood_urea_nitrogen = 7.1
        result = self.judge.judge_high_blood_urea_nitrogen(blood_urea_nitrogen)
        self.assertEqual(1, result['low_protein'])
        self.assertEqual(1, result['blood_urea_nitrogen'])
        blood_urea_nitrogen = 7.0
        result = self.judge.judge_high_blood_urea_nitrogen(blood_urea_nitrogen)
        self.assertEqual(0, result['low_protein'])
        self.assertEqual(0, result['blood_urea_nitrogen'])

    def test_judge_diabetes(self):
        glucose, glycohemoglobin = 7.0, 6.5
        result = self.judge.judge_diabetes(glucose, glycohemoglobin)
        self.assertEqual(1, result['low_sugar'])
        self.assertEqual(1, result['high_fiber'])
        self.assertEqual(1, result['diabetes'])
        glucose, glycohemoglobin = 7.0, 6.4
        result = self.judge.judge_diabetes(glucose, glycohemoglobin)
        self.assertEqual(0, result['low_sugar'])
        self.assertEqual(0, result['high_fiber'])
        self.assertEqual(0, result['diabetes'])
        glucose, glycohemoglobin = 6.9, 6.5
        result = self.judge.judge_diabetes(glucose, glycohemoglobin)
        self.assertEqual(0, result['low_sugar'])
        self.assertEqual(0, result['high_fiber'])
        self.assertEqual(0, result['diabetes'])

    def test_judge_anemia(self):
        red_blood_cell_count, hemoglobin, gender = 4, 13.1, 1
        result = self.judge.judge_anemia(red_blood_cell_count, hemoglobin, gender)
        self.assertEqual(1, result['high_iron'])
        self.assertEqual(1, result['high_vitamin_c'])
        self.assertEqual(1, result['high_folate_acid'])
        self.assertEqual(1, result['high_vitamin_b12'])
        self.assertEqual(1, result['anemia'])
        red_blood_cell_count, hemoglobin, gender = 4, 11.5, 2
        result = self.judge.judge_anemia(red_blood_cell_count, hemoglobin, gender)
        self.assertEqual(1, result['high_iron'])
        self.assertEqual(1, result['high_vitamin_c'])
        self.assertEqual(1, result['high_folate_acid'])
        self.assertEqual(1, result['high_vitamin_b12'])
        self.assertEqual(1, result['anemia'])
        red_blood_cell_count, hemoglobin, gender = 4, 13.2, 1
        result = self.judge.judge_anemia(red_blood_cell_count, hemoglobin, gender)
        self.assertEqual(0, result['high_iron'])
        self.assertEqual(0, result['high_vitamin_c'])
        self.assertEqual(0, result['high_folate_acid'])
        self.assertEqual(0, result['high_vitamin_b12'])
        self.assertEqual(0, result['anemia'])
        red_blood_cell_count, hemoglobin, gender = 4, 11.6, 2
        result = self.judge.judge_anemia(red_blood_cell_count, hemoglobin, gender)
        self.assertEqual(0, result['high_iron'])
        self.assertEqual(0, result['high_vitamin_c'])
        self.assertEqual(0, result['high_folate_acid'])
        self.assertEqual(0, result['high_vitamin_b12'])
        self.assertEqual(0, result['anemia'])
        red_blood_cell_count, hemoglobin, gender = 5, 11.5, 2
        result = self.judge.judge_anemia(red_blood_cell_count, hemoglobin, gender)
        self.assertEqual(0, result['high_iron'])
        self.assertEqual(0, result['high_vitamin_c'])
        self.assertEqual(0, result['high_folate_acid'])
        self.assertEqual(0, result['high_vitamin_b12'])
        self.assertEqual(0, result['anemia'])

    def test_judge_osteoporosis(self):
        osteoporosis = 1
        result = self.judge.judge_osteoporosis(osteoporosis)
        self.assertEqual(1, result['high_calcium'])
        self.assertEqual(1, result['high_vitamin_d'])
        self.assertEqual(1, result['high_vitamin_c'])
        osteoporosis = 0
        result = self.judge.judge_osteoporosis(osteoporosis)
        self.assertEqual(0, result['high_calcium'])
        self.assertEqual(0, result['high_vitamin_d'])
        self.assertEqual(0, result['high_vitamin_c'])

    def test_judge_opioid_misuse_multum_drug_therapeutic_category(self):
        l_one,l_two,l_three=57,58,192
        result = self.judge.judge_opioid_misuse_multum_drug_therapeutic_category(l_one,l_two,l_three)
        self.assertEqual(True,result)
        l_one,l_two,l_three=57,59,192
        result = self.judge.judge_opioid_misuse_multum_drug_therapeutic_category(l_one,l_two,l_three)
        self.assertEqual(False,result)

    def test_judge_opioid_misuse_drug_used(self):
        used=1
        drug_id=1
        multum_drug_therapeutic_categories=[1,2,3]
        icd_code='F11.2'
        drunk_days=91
        result = self.judge.judge_opioid_misuse_drug_used(used, drug_id, multum_drug_therapeutic_categories, icd_code, drunk_days)
        self.assertEqual(True, result)

if __name__ == '__main__':
    unittest.main()
