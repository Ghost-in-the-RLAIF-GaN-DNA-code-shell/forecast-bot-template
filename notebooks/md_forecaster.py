import asyncio
import logging
from datetime import datetime
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    clean_indents,
)
from forecast_bot_template import TemplateForecaster  # adjust import if your file is named differently

logger = logging.getLogger(__name__)


class MidrangeForecaster(TemplateForecaster):
    """
    A practical upgrade over TemplateForecaster:
    - Summarizes raw research into concise, cited bullets (reduces noise).
    - Adds light calibration guards:
        * Binary: small shrink toward 50%.
        * Multiple choice: normalization + small floors per option.
        * Numeric: enforce non-decreasing percentiles.
    - Keeps prompts in-place (no centralized templates), preserving your existing style.
    """

    # Tuning knobs (kept simple and conservative)
    BINARY_SHRINK = 0.2       # 0.0=no shrink; 1.0=force to 0.5
    MC_MIN_FLOOR = 1.0        # min 1% per option before renormalization
    MAX_SUMMARY_CHARS = 900   # keep research concise

    async def _summarize_research(self, research: str, question_text: str) -> str:
        """
        Condense raw research into 4–6 bullets with inline URLs where possible.
        Uses the 'summarizer' LLM if configured, else the default.
        """
        if not research:
            return ""
        try:
            summarizer = self.get_llm("summarizer", "llm")
        except Exception:
            summarizer = self.get_llm("default", "llm")

        prompt = clean_indents(f"""
        You are assisting a professional forecaster.
        Condense the following into 4–6 concise bullet points, each with a parenthetical source URL if present.
        Use only information in the text. No speculation. Keep under 120 words.

        Question:
        {question_text}

        Research:
        {research}
        """)
        try:
            summary = await summarizer.invoke(prompt)
            return summary.strip()[: self.MAX_SUMMARY_CHARS]
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return research[: self.MAX_SUMMARY_CHARS]

    async def run_research(self, question) -> str:  # override to add summarization
        raw = await super().run_research(question)
        summarized = await self._summarize_research(raw or "", question.question_text)
        logger.info(f"Summarized research for {question.page_url}:\n{summarized}")
        return summarized

    # ---------- Binary ----------

    @staticmethod
    def _shrink_toward_half(p: float, shrink: float) -> float:
        """
        Blend probability toward 0.5 by 'shrink' fraction.
        p in [0,1]; shrink in [0,1].
        """
        return 0.5 + (p - 0.5) * (1.0 - max(0.0, min(1.0, shrink)))

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Use the template prompt unchanged
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        pred = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        # Light calibration: shrink toward 0.5 to avoid overconfident swings
        pred = self._shrink_toward_half(pred, self.BINARY_SHRINK)
        logger.info(
            f"[Binary] {question.page_url} -> {pred:.3f}\n{reasoning[:600]}"
        )
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

    # ---------- Multiple choice ----------

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        raw: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )

        # Normalize and apply a small per-option floor, preserving option order
        opts = list(question.options)
        raw_map = {name: float(pct) for name, pct in raw}
        pcts = []
        for o in opts:
            p = max(self.MC_MIN_FLOOR, raw_map.get(o, 0.0))
            pcts.append(p)
        s = sum(pcts) or 1.0
        pcts = [round(100.0 * x / s, 2) for x in pcts]
        pred: PredictedOptionList = list(zip(opts, pcts))  # type: ignore

        logger.info(f"[MC] {question.page_url} -> {pred}\n{reasoning[:600]}")
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

    # ---------- Numeric ----------

    @staticmethod
    def _enforce_monotonic_percentiles(dist: NumericDistribution) -> NumericDistribution:
        """
        Ensure declared_percentiles is non-decreasing (10<=20<=40<=60<=80<=90).
        """
        ps = dist.declared_percentiles
        keys = sorted(ps.keys(), key=lambda k: float(k))
        vals = [float(ps[k]) for k in keys]
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        dist.declared_percentiles = {k: vals[idx] for idx, k in enumerate(keys)}
        return dist

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        dist: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        dist = self._enforce_monotonic_percentiles(dist)
        logger.info(
            f"[Numeric] {question.page_url} -> {dist.declared_percentiles}\n{reasoning[:600]}"
        )
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

#Brief function-by-function summary
#_summarize_research(research, question_text): Condenses raw research into short, relevant bullets to reduce noise before prompting.

#run_research(question): Calls the template’s research providers, then summarizes; returns concise, readable context.

#_shrink_toward_half(p, shrink): Light calibration to avoid overconfident binary predictions.

#_run_forecast_on_binary(...): Reuses the template’s prompt and parsing; adds small shrink toward 50%.

#_run_forecast_on_multiple_choice(...): Normalizes option probabilities and applies a minimal per-option floor; preserves option order.

#_enforce_monotonic_percentiles(dist): Makes numeric percentiles non-decreasing to avoid malformed distributions.

#_run_forecast_on_numeric(...): Reuses the template’s prompt and parsing; applies monotonicity guard.
