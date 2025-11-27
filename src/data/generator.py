import random
import textwrap
from typing import List, Dict, Optional
from faker import Faker
from datasets import load_dataset

class BaseGenerator:
    def generate(self, n_docs: int, context_tokens: int, positions: List[str]) -> List[Dict]:
        raise NotImplementedError

    def _add_offsets(self, dataset: List[Dict]) -> List[Dict]:
        """Helper to add answer start/end offsets to the dataset items."""
        for item in dataset:
            doc = item["document"]
            answer = item["answer"]
            # Find the answer in the document
            start_idx = doc.find(answer)
            if start_idx != -1:
                item["answer_start_index"] = start_idx
                item["answer_end_index"] = start_idx + len(answer)
            else:
                # Should not happen given how we generate, but safety first
                item["answer_start_index"] = -1
                item["answer_end_index"] = -1
        return dataset

class Tier1Synthetic(BaseGenerator):
    def __init__(self):
        self.fake = Faker()

    def _generate_text(self, target_tokens: int) -> str:
        # Faker generates words, approx 1.3 tokens per word is a rough heuristic, 
        # but let's just generate text and trim.
        # Generating huge text with faker can be slow, so we generate paragraphs.
        text = ""
        while len(text.split()) < target_tokens:
            text += self.fake.paragraph(nb_sentences=5) + " " + self.fake.bs() + ". "
            if random.random() < 0.3:
                text += f"The entity {self.fake.company()} located in {self.fake.city()} reported {self.fake.bs()}. "
        
        words = text.split()
        return " ".join(words[:target_tokens])

    def generate(self, n_docs: int, context_tokens: int, positions: List[str]) -> List[Dict]:
        dataset = []
        for _ in range(n_docs):
            position = random.choice(positions)
            answer_code = str(random.randint(1000, 9999))
            answer = f"ANSWER-{answer_code}"
            
            # We need to insert the answer at the right position
            # Total text needed approx context_tokens
            
            # Simple strategy: generate 3 chunks
            chunk_size = context_tokens // 3
            
            part1 = self._generate_text(chunk_size)
            part2 = self._generate_text(chunk_size)
            part3 = self._generate_text(chunk_size)
            
            if position == "start":
                doc = f"{answer} {part1} {part2} {part3}"
            elif position == "end":
                doc = f"{part1} {part2} {part3} {answer}"
            else: # middle
                doc = f"{part1} {answer} {part2} {part3}"
            
            doc = textwrap.fill(doc, width=120)
            
            dataset.append({
                "document": doc,
                "question": "What is the hidden code? Respond with the exact code.",
                "answer": answer,
                "position": position,
                "context_tokens": context_tokens, # Approximate
                "tier": "Tier 1 (Synthetic)"
            })
        return self._add_offsets(dataset)

class Tier2Real(BaseGenerator):
    def __init__(self):
        # Load wikitext-2-v1, streaming is better for speed if we don't need it all,
        # but for random access, loading a split is easier. It's small (few MBs).
        try:
            self.dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
            self.text_pool = [x['text'] for x in self.dataset if len(x['text']) > 100]
        except Exception as e:
            print(f"Error loading wikitext: {e}. Fallback to synthetic.")
            self.fallback = Tier1Synthetic()
        else:
            self.fallback = None

    def _get_real_text(self, target_tokens: int) -> str:
        if self.fallback:
            return self.fallback._generate_text(target_tokens)
        
        # Concatenate random articles until we have enough text
        text = ""
        while len(text.split()) < target_tokens:
            text += random.choice(self.text_pool) + "\n"
        
        words = text.split()
        return " ".join(words[:target_tokens])

    def generate(self, n_docs: int, context_tokens: int, positions: List[str]) -> List[Dict]:
        dataset = []
        for _ in range(n_docs):
            position = random.choice(positions)
            answer_code = str(random.randint(1000, 9999))
            answer = f"ANSWER-{answer_code}"
            
            chunk_size = context_tokens // 3
            part1 = self._get_real_text(chunk_size)
            part2 = self._get_real_text(chunk_size)
            part3 = self._get_real_text(chunk_size)
            
            if position == "start":
                doc = f"{answer}\n\n{part1}\n{part2}\n{part3}"
            elif position == "end":
                doc = f"{part1}\n{part2}\n{part3}\n\n{answer}"
            else:
                doc = f"{part1}\n\n{answer}\n\n{part2}\n{part3}"
            
            # Cleanup multiple newlines
            doc = "\n".join([line.strip() for line in doc.splitlines() if line.strip()])
            
            dataset.append({
                "document": doc,
                "question": "What is the hidden code? Respond with the exact code.",
                "answer": answer,
                "position": position,
                "context_tokens": context_tokens,
                "tier": "Tier 2 (Real)"
            })
        return self._add_offsets(dataset)

class Tier3Adversarial(Tier2Real):
    def generate(self, n_docs: int, context_tokens: int, positions: List[str]) -> List[Dict]:
        # Generate base items from Tier 2
        base_items = super().generate(n_docs, context_tokens, positions)
        
        for item in base_items:
            item["tier"] = "Tier 3 (Adversarial)"
            true_answer = item["answer"]
            true_code = true_answer.split("-")[1]
            
            # Create a distractor
            distractor_code = str(int(true_code) + 1) # Close number
            distractor = f"ANSWER-{distractor_code}"
            
            # Inject distractor at a different position
            # If answer is at start, put distractor at end, etc.
            # We'll just append/prepend or insert it into the text.
            
            doc = item["document"]
            words = doc.split()
            
            # Simple injection strategy for now:
            # If position is start, put distractor in middle or end
            # We don't want to overwrite the true answer.
            
            # Let's just insert it at a random location that is NOT the true answer location
            # But we need to be careful not to make the distractor the "first" answer if the model just picks the first one?
            # The prompt says "exact code".
            # Let's add a "near-miss" sentence.
            
            distractor_sentence = f" The code for the previous section was ANSWER-{distractor_code}. "
            
            # Insert randomly
            insert_idx = random.randint(0, len(words))
            words.insert(insert_idx, distractor_sentence)
            
            item["document"] = " ".join(words)
            
        # Recalculate offsets because document changed
        return self._add_offsets(base_items)

class Tier4MultiHop(Tier2Real):
    def generate(self, n_docs: int, context_tokens: int, positions: List[str]) -> List[Dict]:
        dataset = []
        for _ in range(n_docs):
            # We need 2 facts.
            # Fact 1: "The secret code is X."
            # Fact 2: "The operation is Y."
            # Question: "What is the result of operation Y on code X?" (Requires reasoning)
            
            # Or simpler: "What is the code mentioned in the section about [Topic A]?"
            # But that's just retrieval.
            
            # Let's do:
            # Fact A: "Project Alpha's code is 1234."
            # Fact B: "Project Beta's code is 5678."
            # Question: "What is the sum of Project Alpha and Project Beta codes?"
            
            code_a = random.randint(100, 999)
            code_b = random.randint(100, 999)
            sum_code = code_a + code_b
            
            fact_a = f"The security code for Project Alpha is {code_a}."
            fact_b = f"The security code for Project Beta is {code_b}."
            
            # Generate background text
            chunk_size = context_tokens // 3
            part1 = self._get_real_text(chunk_size)
            part2 = self._get_real_text(chunk_size)
            part3 = self._get_real_text(chunk_size)
            
            # Insert facts at random positions
            # We have 3 chunks. We can put facts in between or inside.
            # Let's put Fact A in Part 1 and Fact B in Part 3 (long range dependency).
            
            # Inject into text
            words1 = part1.split()
            words1.insert(random.randint(0, len(words1)), fact_a)
            part1 = " ".join(words1)
            
            words3 = part3.split()
            words3.insert(random.randint(0, len(words3)), fact_b)
            part3 = " ".join(words3)
            
            doc = f"{part1}\n\n{part2}\n\n{part3}"
            
            dataset.append({
                "document": doc,
                "question": "What is the sum of the security codes for Project Alpha and Project Beta? Respond with just the number.",
                "answer": str(sum_code),
                "position": "multi-hop", # Special position
                "context_tokens": context_tokens,
                "tier": "Tier 4 (Multi-hop)"
            })
        return self._add_offsets(dataset)

def get_generator(tier: str):
    if tier.startswith("Tier 1"):
        return Tier1Synthetic()
    elif tier.startswith("Tier 2"):
        return Tier2Real()
    elif tier.startswith("Tier 3"):
        return Tier3Adversarial()
    elif tier.startswith("Tier 4"):
        return Tier4MultiHop()
    else:
        return Tier1Synthetic()
