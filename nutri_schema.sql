--
-- PostgreSQL database dump
--

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.6

-- Started on 2025-08-29 15:13:26

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 217 (class 1259 OID 16738)
-- Name: daily_food_with_nutrition_target_bronze; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.daily_food_with_nutrition_target_bronze (
    daily_food_id text,
    target text,
    user_id text,
    grams numeric,
    calorie numeric,
    protein numeric,
    carb numeric,
    sugar numeric,
    fiber numeric,
    saturated_fat numeric,
    cholesterol numeric,
    folic_acid numeric,
    vitamin_b12 numeric,
    vitamin_c numeric,
    vitamin_d numeric,
    calcium numeric,
    phosphorus numeric,
    potassium numeric,
    iron numeric,
    sodium numeric,
    age_group numeric,
    gender numeric,
    age integer,
    user_low_phosphorus integer,
    user_low_carb integer,
    weight numeric,
    height numeric,
    under_weight integer,
    over_weight integer,
    user_low_calorie integer,
    user_high_calorie integer,
    user_low_sodium integer,
    blood_pressure integer,
    user_high_potassium integer,
    user_low_saturated_fat integer,
    user_low_cholesterol integer,
    low_density_lipoprotein numeric,
    blood_urea_nitrogen numeric,
    user_low_protein integer,
    user_high_protein integer,
    opioid_misuse integer,
    user_low_sugar integer,
    user_high_fiber integer,
    diabetes integer,
    user_high_folate_acid integer,
    user_high_iron integer,
    user_high_vitamin_b12 integer,
    anemia integer,
    osteoporosis integer,
    user_high_calcium integer,
    user_high_vitamin_c integer,
    user_high_vitamin_d integer,
    level numeric,
    match boolean
);


ALTER TABLE public.daily_food_with_nutrition_target_bronze OWNER TO postgres;

--
-- TOC entry 218 (class 1259 OID 16743)
-- Name: daily_food_with_nutrition_target_gold; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.daily_food_with_nutrition_target_gold (
    user_id text,
    age_group numeric,
    gender numeric,
    grams numeric,
    calorie numeric,
    protein numeric,
    carb numeric,
    sugar numeric,
    fiber numeric,
    saturated_fat numeric,
    cholesterol numeric,
    folic_acid numeric,
    vitamin_b12 numeric,
    vitamin_c numeric,
    vitamin_d numeric,
    calcium numeric,
    phosphorus numeric,
    potassium numeric,
    iron numeric,
    sodium numeric,
    age numeric,
    user_low_phosphorus integer,
    user_low_carb integer,
    weight numeric,
    height numeric,
    under_weight integer,
    over_weight integer,
    user_low_calorie integer,
    user_high_calorie integer,
    user_low_sodium integer,
    blood_pressure integer,
    user_high_potassium integer,
    user_low_saturated_fat integer,
    user_low_cholesterol integer,
    low_density_lipoprotein numeric,
    blood_urea_nitrogen numeric,
    user_low_protein integer,
    user_high_protein integer,
    opioid_misuse integer,
    user_low_sugar integer,
    user_high_fiber integer,
    diabetes integer,
    user_high_folate_acid integer,
    user_high_iron integer,
    user_high_vitamin_b12 integer,
    anemia integer,
    osteoporosis integer,
    user_high_calcium integer,
    user_high_vitamin_c integer,
    user_high_vitamin_d integer,
    level numeric,
    b_calorie boolean,
    b_carb boolean,
    b_fiber boolean,
    b_protein boolean,
    b_saturated_fat boolean,
    b_sugar boolean,
    b_cholesterol boolean,
    macro_health_score integer,
    b_sodium boolean,
    b_phosphorus boolean,
    b_potassium boolean,
    b_iron boolean,
    b_calcium boolean,
    b_folic_acid boolean,
    b_vitamin_c boolean,
    b_vitamin_d boolean,
    b_vitamin_b12 boolean,
    micro_health_score integer,
    match boolean,
    total_positive_score integer,
    total_negative_score integer,
    daily_food_id text,
    target text
);


ALTER TABLE public.daily_food_with_nutrition_target_gold OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 16748)
-- Name: daily_food_with_nutrition_target_silver; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.daily_food_with_nutrition_target_silver (
    user_id text,
    gender numeric,
    age integer,
    user_low_phosphorus integer,
    user_low_carb integer,
    weight numeric,
    height numeric,
    under_weight integer,
    over_weight integer,
    user_low_calorie integer,
    user_high_calorie integer,
    user_low_sodium integer,
    blood_pressure integer,
    user_high_potassium integer,
    user_low_saturated_fat integer,
    user_low_cholesterol integer,
    low_density_lipoprotein numeric,
    blood_urea_nitrogen numeric,
    user_low_protein integer,
    user_high_protein integer,
    opioid_misuse integer,
    user_low_sugar integer,
    user_high_fiber integer,
    diabetes integer,
    user_high_folate_acid integer,
    user_high_iron integer,
    user_high_vitamin_b12 integer,
    anemia integer,
    osteoporosis integer,
    user_high_calcium integer,
    user_high_vitamin_c integer,
    user_high_vitamin_d integer,
    level numeric,
    target text,
    age_group numeric,
    grams numeric,
    calorie numeric,
    protein numeric,
    carb numeric,
    sugar numeric,
    fiber numeric,
    saturated_fat numeric,
    cholesterol numeric,
    folic_acid numeric,
    vitamin_b12 numeric,
    vitamin_c numeric,
    vitamin_d numeric,
    calcium numeric,
    phosphorus numeric,
    potassium numeric,
    iron numeric,
    sodium numeric,
    b_calorie text,
    b_carb text,
    b_fiber text,
    b_protein text,
    b_saturated_fat text,
    b_sugar text,
    b_cholesterol text,
    b_sodium text,
    b_phosphorus text,
    b_potassium text,
    b_iron text,
    b_calcium text,
    b_folic_acid text,
    b_vitamin_c text,
    b_vitamin_d text,
    b_vitamin_b12 text,
    match boolean,
    total_positive_score integer,
    total_negative_score integer,
    daily_food_id text,
    macro_health_score integer,
    micro_health_score integer
);


ALTER TABLE public.daily_food_with_nutrition_target_silver OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 16753)
-- Name: food_code; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.food_code (
    food_id text NOT NULL,
    food_desc text,
    food_desc_long text,
    years text,
    positive integer,
    negative integer
);


ALTER TABLE public.food_code OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 16758)
-- Name: food_user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.food_user (
    food_id text,
    user_id text,
    eating_type numeric,
    grams numeric,
    day numeric,
    years text,
    daily_food_id text
);


ALTER TABLE public.food_user OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 16783)
-- Name: meal_plan; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.meal_plan (
    daily_food_id text,
    target text,
    grams numeric,
    calorie numeric,
    protein numeric,
    carb numeric,
    sugar numeric,
    fiber numeric,
    saturated_fat numeric,
    cholesterol numeric,
    folic_acid numeric,
    vitamin_b12 numeric,
    vitamin_c numeric,
    vitamin_d numeric,
    calcium numeric,
    phosphorus numeric,
    potassium numeric,
    iron numeric,
    sodium numeric,
    level text
);


ALTER TABLE public.meal_plan OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 16763)
-- Name: targets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.targets (
    target text NOT NULL,
    calorie_min numeric,
    calorie_max numeric,
    protein_min numeric,
    protein_max numeric,
    saturated_fat_min numeric,
    saturated_fat_max numeric,
    carb_min numeric,
    carb_max numeric,
    fiber_min numeric,
    fiber_max numeric,
    sugar_min numeric,
    sugar_max numeric,
    cholesterol_min numeric,
    cholesterol_max numeric,
    sodium_min numeric,
    sodium_max numeric,
    calcium_min numeric,
    calcium_max numeric,
    phosphorus_min numeric,
    phosphorus_max numeric,
    potassium_min numeric,
    potassium_max numeric,
    iron_min numeric,
    iron_max numeric,
    folic_acid_min numeric,
    folic_acid_max numeric,
    vitamin_c_min numeric,
    vitamin_c_max numeric,
    vitamin_d_min numeric,
    vitamin_d_max numeric,
    vitamin_b12_min numeric,
    vitamin_b12_max numeric
);


ALTER TABLE public.targets OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 16778)
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    user_id text NOT NULL,
    gender numeric,
    age integer,
    weight numeric,
    height numeric,
    level numeric,
    under_weight integer,
    over_weight integer,
    blood_pressure integer,
    low_density_lipoprotein numeric,
    blood_urea_nitrogen numeric,
    opioid_misuse integer,
    diabetes integer,
    anemia integer,
    osteoporosis integer,
    user_low_phosphorus integer,
    user_low_carb integer,
    user_low_calorie integer,
    user_high_calorie integer,
    user_low_sodium integer,
    user_high_potassium integer,
    user_low_saturated_fat integer,
    user_low_cholesterol integer,
    user_low_protein integer,
    user_high_protein integer,
    user_low_sugar integer,
    user_high_fiber integer,
    user_high_folate_acid integer,
    user_high_iron integer,
    user_high_vitamin_b12 integer,
    user_high_calcium integer,
    user_high_vitamin_c integer,
    user_high_vitamin_d integer,
    years text
);


ALTER TABLE public.users OWNER TO postgres;


--
-- TOC entry 4672 (class 2606 OID 16769)
-- Name: food_code food_code_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.food_code
    ADD CONSTRAINT food_code_pkey PRIMARY KEY (food_id);


--
-- TOC entry 4677 (class 2606 OID 16771)
-- Name: targets targets_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.targets
    ADD CONSTRAINT targets_pkey PRIMARY KEY (target);


--
-- TOC entry 4673 (class 1259 OID 16772)
-- Name: idx_food_code_food_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_food_code_food_id ON public.food_code USING btree (food_id);


--
-- TOC entry 4674 (class 1259 OID 16773)
-- Name: idx_food_user_daily_food_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_food_user_daily_food_id ON public.food_user USING btree (daily_food_id);


--
-- TOC entry 4675 (class 1259 OID 16774)
-- Name: idx_food_user_food_id_unique; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_food_user_food_id_unique ON public.food_user USING btree (food_id);


--
-- TOC entry 4668 (class 1259 OID 16775)
-- Name: idx_target_text_bronze; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_target_text_bronze ON public.daily_food_with_nutrition_target_bronze USING btree (target);


--
-- TOC entry 4669 (class 1259 OID 16776)
-- Name: idx_target_text_gold; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_target_text_gold ON public.daily_food_with_nutrition_target_gold USING btree (target);


--
-- TOC entry 4670 (class 1259 OID 16777)
-- Name: idx_target_text_silver; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_target_text_silver ON public.daily_food_with_nutrition_target_silver USING btree (target);


-- Completed on 2025-08-29 15:13:26

--
-- PostgreSQL database dump complete
--
