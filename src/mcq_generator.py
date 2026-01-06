"""
MCQ Quiz Generator - OpenRouter Version (FREE!)
"""

import os
import json
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class MCQGenerator:
    """Generate MCQ quizzes from course content using OpenRouter"""
    
    def __init__(self, rag_system):
        """Initialize with RAG system"""
        self.rag = rag_system
        
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        self.llm = OpenAI(
            temperature=0.7,
            model_name="openai/gpt-3.5-turbo-instruct",
            openai_api_key=api_key,
            openai_api_base=api_base,
            max_tokens=1000
        )
    
    def generate_quiz(self, topic: str, num_questions: int = 5):
        """Generate MCQ quiz on a topic"""
        
        print(f"\n🎯 Generating {num_questions} MCQs on: {topic}")
        
        # Get relevant content from course materials
        print("📚 Retrieving relevant content...")
        result = self.rag.ask(f"Explain {topic} in detail with key concepts")
        context = result['answer']
        
        # Add source content
        source_texts = [doc['content'] for doc in result['sources']]
        full_context = context + "\n\n" + "\n".join(source_texts)
        
        # Create prompt for MCQ generation
        prompt_template = """Based on the following course content about {topic}, generate {num_questions} multiple-choice questions.

Course Content:
{context}

Generate EXACTLY {num_questions} questions in this JSON format:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "First option",
      "B": "Second option", 
      "C": "Third option",
      "D": "Fourth option"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation why this is correct"
  }}
]

Requirements:
- Test understanding, not just memorization
- All options should be plausible
- Include brief explanations
- Focus on the key concepts

Generate the questions:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["topic", "num_questions", "context"]
        )
        
        # Generate MCQs
        print("🤖 Generating questions...")
        formatted_prompt = prompt.format(
            topic=topic,
            num_questions=num_questions,
            context=full_context[:3000]  # Limit context size
        )
        
        response = self.llm(formatted_prompt)
        
        try:
            # Parse JSON response
            mcqs = json.loads(response)
            print(f"✅ Generated {len(mcqs)} questions!")
            return mcqs
        except json.JSONDecodeError:
            print("⚠️  Could not parse JSON, creating fallback quiz...")
            return self._create_fallback_quiz(topic, num_questions)
    
    def _create_fallback_quiz(self, topic, num_questions):
        """Create a simple fallback quiz if JSON parsing fails"""
        return [{
            "question": f"What is a key concept in {topic}?",
            "options": {
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            },
            "correct_answer": "A",
            "explanation": "This is a sample question. Try regenerating the quiz."
        }] * num_questions
    
    def format_quiz(self, mcqs, topic):
        """Format quiz for display"""
        output = []
        output.append("="*70)
        output.append(f"📝 QUIZ: {topic.upper()}")
        output.append("="*70)
        
        for i, mcq in enumerate(mcqs, 1):
            output.append(f"\n{'─'*70}")
            output.append(f"Question {i}: {mcq['question']}")
            output.append("")
            
            for key, value in mcq['options'].items():
                output.append(f"  {key}) {value}")
            
            output.append(f"\n✓ Correct Answer: {mcq['correct_answer']}")
            output.append(f"📖 Explanation: {mcq['explanation']}")
        
        output.append("\n" + "="*70)
        return "\n".join(output)
    
    def save_quiz(self, mcqs, topic, filename=None):
        """Save quiz to file"""
        if not filename:
            filename = f"outputs/{topic.replace(' ', '_')}_quiz.txt"
        
        os.makedirs("outputs", exist_ok=True)
        
        formatted_quiz = self.format_quiz(mcqs, topic)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(formatted_quiz)
        
        print(f"\n💾 Quiz saved to: {filename}")
        return filename

# Test the generator
if __name__ == "__main__":
    from rag_system import RAGSystem
    
    print("\n" + "="*70)
    print("🎓 MCQ QUIZ GENERATOR TEST - OpenRouter Version")
    print("="*70)
    
    # Initialize systems
    print("\n1. Initializing RAG system...")
    rag = RAGSystem()
    rag.initialize()
    
    print("\n2. Initializing quiz generator...")
    quiz_gen = MCQGenerator(rag)
    
    # Generate quiz
    topic = "Convolutional Neural Networks"
    num_questions = 3
    
    print(f"\n3. Generating quiz...")
    mcqs = quiz_gen.generate_quiz(topic, num_questions)
    
    # Display and save
    print("\n" + quiz_gen.format_quiz(mcqs, topic))
    quiz_gen.save_quiz(mcqs, topic)
    
    print("\n✅ Test complete!")