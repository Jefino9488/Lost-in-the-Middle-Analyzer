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
            
            # Generate multiple semantic distractor types
            distractors = []
            distractor_sentences = []
            
            # Type 1: Close number variants (+/-1)
            close_code_plus = str(int(true_code) + 1)
            close_code_minus = str(max(1000, int(true_code) - 1))
            distractors.append(f"ANSWER-{close_code_plus}")
            distractors.append(f"ANSWER-{close_code_minus}")
            
            # Type 2: Digit transposition (1234 -> 1243)
            if len(true_code) >= 2:
                digits = list(true_code)
                # Swap last two digits
                digits[-1], digits[-2] = digits[-2], digits[-1]
                transposed_code = ''.join(digits)
                distractors.append(f"ANSWER-{transposed_code}")
            
            # Type 3: Create contextual paraphrases
            distractor_sentences = [
                f"The preliminary access code was {distractors[0]}.",
                f"Prior versions used the identifier {distractors[1]}.",
                f"The deprecated code {distractors[2] if len(distractors) > 2 else distractors[0]} should not be used.",
                f"During testing, we encountered {distractors[0]}, but this was incorrect.",
                f"The system previously generated {distractors[1]} before the update."
            ]
            
            doc = item["document"]
            words = doc.split()
            
            # Insert 2-3 random distractors at different positions
            n_distractors = random.randint(2, 3)
            selected_sentences = random.sample(distractor_sentences, n_distractors)
            
            for distractor_sent in selected_sentences:
                # Find a position far from true answer
                true_pos = doc.find(true_answer)
                doc_len = len(doc)
                
                # Try to place distractor in opposite third of document
                if true_pos < doc_len // 3:
                    # True answer in first third, place distractor in last third
                    insert_range = (2 * len(words) // 3, len(words))
                elif true_pos > 2 * doc_len // 3:
                    # True answer in last third, place distractor in first third
                    insert_range = (0, len(words) // 3)
                else:
                    # True answer in middle, place at edges
                    insert_range = (0, len(words) // 4) if random.random() < 0.5 else (3 * len(words) // 4, len(words))
                
                insert_idx = random.randint(max(0, insert_range[0]), min(len(words), insert_range[1]))
                words.insert(insert_idx, f" {distractor_sent} ")
            
            item["document"] = " ".join(words)
        
        # Recalculate offsets because document changed
        return self._add_offsets(base_items)

class Tier4MultiHop(Tier2Real):
    def generate(self, n_docs: int, context_tokens: int, positions: List[str]) -> List[Dict]:
        dataset = []
        
        # Different reasoning task types
        task_types = ["coreference", "temporal", "compositional"]
        
        for _ in range(n_docs):
            task_type = random.choice(task_types)
            chunk_size = context_tokens // 4  # Use 4 chunks for more spacing
            
            if task_type == "coreference":
                # Coreference resolution: "Company A's code is X. They also have Y. What is A's first code?"
                company = random.choice(["Acme Corp", "GlobalTech", "Innovate Inc", "TechVision"])
                code_a = random.randint(1000, 9999)
                code_b = random.randint(1000, 9999)
                
                fact1 = f"The primary access code for {company} is ANSWER-{code_a}."
                fact2 = f"They also maintain a secondary code of ANSWER-{code_b} for backup systems."
                distractor = f"A different company uses ANSWER-{random.randint(1000, 9999)} for their operations."
                
                # Generate background
                part1 = self._get_real_text(chunk_size)
                part2 = self._get_real_text(chunk_size)
                part3 = self._get_real_text(chunk_size)
                part4 = self._get_real_text(chunk_size)
                
                # Inject facts far apart
                words1 = part1.split()
                words1.insert(random.randint(len(words1)//2, len(words1)), fact1)  # Later in first chunk
                part1 = " ".join(words1)
                
                words3 = part3.split()
                words3.insert(random.randint(0, len(words3)//2), fact2)  # Earlier in third chunk
                part3 = " ".join(words3)
                
                # Add distractor in middle
                words2 = part2.split()
                words2.insert(random.randint(0, len(words2)), distractor)
                part2 = " ".join(words2)
                
                doc = f"{part1}\n\n{part2}\n\n{part3}\n\n{part4}"
                question = f"What is the primary access code for {company}? Respond with the exact code."
                answer = f"ANSWER-{code_a}"
                
            elif task_type == "temporal":
                # Temporal reasoning: "Code changed from X to Y in 2020. What was it before 2020?"
                old_code = random.randint(1000, 9999)
                new_code = random.randint(1000, 9999)
                year = random.randint(2018, 2022)
                
                fact1 = f"Before {year}, the system access code was ANSWER-{old_code}."
                fact2 = f"In {year}, security protocols were updated and the new code became ANSWER-{new_code}."
                distractor = f"Another system has used ANSWER-{random.randint(1000, 9999)} since {year-3}."
                
                part1 = self._get_real_text(chunk_size)
                part2 = self._get_real_text(chunk_size)
                part3 = self._get_real_text(chunk_size)
                part4 = self._get_real_text(chunk_size)
                
                # Place temporal facts in different sections
                words2 = part2.split()
                words2.insert(random.randint(0, len(words2)), fact1)
                part2 = " ".join(words2)
                
                words4 = part4.split()
                words4.insert(random.randint(0, len(words4)), fact2)
                part4 = " ".join(words4)
                
                words1 = part1.split()
                words1.insert(random.randint(0, len(words1)), distractor)
                part1 = " ".join(words1)
                
                doc = f"{part1}\n\n{part2}\n\n{part3}\n\n{part4}"
                question = f"What was the system access code before {year}? Respond with the exact code."
                answer = f"ANSWER-{old_code}"
                
            else:  # compositional
                # Compositional: "Project needs codes A and B. A is PREFIX, B is SUFFIX. What is complete code?"
                prefix = random.randint(10, 99)
                suffix = random.randint(10, 99)
                complete_code = f"{prefix}{suffix}"
                
                fact1 = f"The system prefix for authentication is {prefix}."
                fact2 = f"The security suffix code is {suffix}."
                fact3 = f"Complete access requires combining the authentication prefix with the security suffix."
                distractor = f"Legacy systems used prefix {random.randint(10, 99)} which is now deprecated."
                
                part1 = self._get_real_text(chunk_size)
                part2 = self._get_real_text(chunk_size)
                part3 = self._get_real_text(chunk_size)
                part4 = self._get_real_text(chunk_size)
                
                # Distribute facts across document
                words1 = part1.split()
                words1.insert(random.randint(0, len(words1)), fact1)
                part1 = " ".join(words1)
                
                words2 = part2.split()
                words2.insert(random.randint(0, len(words2)), distractor)
                part2 = " ".join(words2)
                
                words3 = part3.split()
                words3.insert(random.randint(0, len(words3)), fact2)
                part3 = " ".join(words3)
                
                words4 = part4.split()
                words4.insert(random.randint(0, len(words4)), fact3)
                part4 = " ".join(words4)
                
                doc = f"{part1}\n\n{part2}\n\n{part3}\n\n{part4}"
                question = "What is the complete access code (prefix + suffix)? Respond with just the numbers."
                answer = complete_code
            
            dataset.append({
                "document": doc,
                "question": question,
                "answer": answer,
                "position": f"multi-hop-{task_type}",
                "context_tokens": context_tokens,
                "tier": "Tier 4 (Multi-hop)",
                "reasoning_type": task_type
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
