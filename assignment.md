# **Project Setup: Running the AI Trip Planner**

This is our main goal for the first session. We are going to get the AI Trip Planner application running locally.

1. [Clone the repo and navigate to the repo](https://www.notion.so/Lesson-1-Building-an-Agent-2693554ef57c80aca00ce1e764da6ae1?pvs=21)
    1. **Navigate into the Project Folder:**
        - `cd ai-trip-planner`
2. **Create the Environment File for Your API Keys:**
    
    This is a crucial step for telling the application what your secret API key is.
    
    - Navigate into the backend directory by typing this into the cursor terminal:
        
        `cd backend`
        
    - Create a new file named `.env`:
        - In your Cursor terminal, type: `touch .env`
    - **Open the new `.env` file** in the Cursor editor (you'll see it in the file directory on the left).
    - Add the following line to the file, replacing `YOUR_API_KEY_HERE` with your actual OpenAI key:
        
        `OPENAI_API_KEY=YOUR_API_KEY_HERE`
        
    - Save the `.env` file. Cursor will automatically use this to load your key.
3. **Set Up the Backend:**
    - Navigate into the project folder: `cd ai-trip-planner`
    - [Create and activate](https://www.notion.so/Lesson-1-Building-an-Agent-2693554ef57c80aca00ce1e764da6ae1?pvs=21) your python virtual environment (if you haven’t already): `source venv/bin/activate`
    - Install the required Python packages: `pip install -r requirements.txt`
4. **Start the Backend Server:**
    - While in backend folder, in the terminal, type in
        - `python main.py`
        - Ignore warnings about arize credentials missing
5. **Set Up and Start the Frontend:**
    - Open a new terminal instance. (click the + button in the top right of the terminal)
    - Navigate into the `frontend` directory: `cd frontend`
    - Install the required packages: in the terminal `npm install`
    - Start the frontend server: `npm start`
6. **View Your Live App!**
    - Your terminal will give you a URL (usually `http://localhost:3000`) to view the running application in your browser.