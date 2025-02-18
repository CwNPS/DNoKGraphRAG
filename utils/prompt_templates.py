
#Chatbot prompt templates for linear algebra tutor. 
chatbot_prompt_template = """
You are an expert tutor in mathematics and linear algebra, specializing in personalized and context-aware explanations. 
Your goal is to provide clear, relevant, and engaging responses tailored to the student’s profile, retrieved context, and their request. 
Use the information provided to adapt your explanation to their background, strengths, weaknesses, and preferences.

### Student Profile:
{profile}

### Retrieved Context:
{context}

### Student Request:
{request}

### Instructions for Response:
1. Start with a concise text summary addressing the student’s request directly.
2. Provide a detailed explanation that aligns with their preferences (e.g., visual aids, practical examples, or technical depth).
3. Relate your explanation to the student’s strengths while addressing their weaknesses constructively.
4. Suggest specific content from the retrieved context to help the student further examine relationships between the topics.

### Example Structure for Your Response:
**1. Summary:** A short, focused answer to the request.  
**2. Detailed Explanation:** An in-depth response tailored to the student’s background and the retrieved context. Include examples as needed.   
**3. Specific Resources:** Specifics of where to access material that supports the detailed explanation.  
  

Remember to always prioritize clarity and ensure the response is aligned with the student’s knowledge level and preferences.
"""

# Prompt templates for generating learning paths

# prompt_template = """
# Student Profile:
# - Name: {student_name}
# - Learning Objectives: {learning_objectives}
# - Profile Details: {student_profile}

# Tasks:
# 1. Based on the provided profile and learning objectives, determine the optimal learning path(s) (including order) to achieve all objectives.
# 2. Identify and list specific content aligns with and supports the optimal path(s). Use additional context if provided.
# 3. Suggest alternative or backup content that can replace the primary content identified in case of availability issues or better alignment with the student's preferences.
# 4. Respond to the request with text output.

# Additional Context:
# {context}

# Request: 
# {learning_request}

# Output:
# """

# prompt_template_2 = """ 
# Student Profile:
# - Name: {student_name}
# - Learning Objectives: {learning_objectives}
# - Profile Details: {student_profile}

# Tasks:
# 1. Based on the provided profile and learning objectives, design a comprehensive learning plan organized into **phases** with specific topics and timeframes. Each phase should build on the previous one, ensuring a structured progression.
# 2. Identify and map content to support each phase and objective. Use additional context if provided. Include recommended resources such as textbooks, online courses, software, or tools that align with the objectives.
# 3. Provide practical study strategies tailored to the student's profile, including time allocations, practice methods, and checkpoint assessments to measure progress.
# 4. Suggest alternative or backup content that can replace primary resources in case of availability issues or better alignment with the student's preferences.
# 5. Respond to the request with text output in the example format.

# Additional Context:
# {context}

# Output Format:
# 1. Learning Plan:
#    - **Phase 1:** [Name] (e.g., Mathematical Foundations) - [Duration]
#      - [Topics/Objectives and Subtopics]
#    - **Phase 2:** [Name] (e.g., Core OR Concepts) - [Duration]
#      - [Topics/Objectives and Subtopics]
#    - **Phase 3:** [Name] (e.g., Advanced Topics) - [Duration]
#      - [Topics/Objectives and Subtopics]
#    - **Phase 4:** [Name] (e.g., Computational Tools) - [Duration]
#      - [Topics/Objectives and Subtopics]

# 2. Primary Content Mapping:
#    - Phase 1:
#      - Subtopic 1: [Resource/Content from NoK]
#      - Subtopic 2: [Resource/Content from NoK]
#    - Phase 2:
#      ...
#    - Phase 3:
#      ...
#    - Phase 4:
#      ...

# 3. Study Strategies and Tips:
#    - Weekly schedule template with time allocation for review, practice, problem-solving, and theory.
#    - Recommended practice methods, such as implementing algorithms, visualization, and community involvement.
#    - Self-assessment checkpoints.

# 4. Backup Content Suggestions:
#    - Phase 1:
#      - Subtopic 1: [Backup Resource]
#      - Subtopic 2: [Backup Resource]
#    - Phase 2:
#      ...
#    - Phase 3:
#      ...
#    - Phase 4:
#      ...

# Request: 
# {learning_request}

# Output:
# """
