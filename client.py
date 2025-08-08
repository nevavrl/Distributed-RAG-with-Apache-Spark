# rag_socket_client.py
import socket
import json
import time

# =========================================================
# Client Configuration
# =========================================================
SERVER_HOST = "localhost"
SERVER_PORT = 8888

class RAGSocketClient:
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
    
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            
            print("Connected to server")
            
            # Clear welcome message
            self._receive_response()
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_query(self, query):
        if not self.is_connected:
            print("Not connected")
            return False
        
        try:
            self.socket.send((query + "\n").encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            return False
    
    def _receive_response(self):
        buffer = ""
        
        while True:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                
                if not data:
                    break
                
                buffer += data
                
                # Process complete JSON messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            response = json.loads(line.strip())
                            return response
                        except json.JSONDecodeError:
                            continue
            
            except Exception as e:
                print(f"Receive error: {e}")
                break
        
        return None
    
    def wait_for_answer(self):
        """Wait for query response"""
        while True:
            response = self._receive_response()
            
            if not response:
                print("Connection lost")
                return False
            
            # Skip non-answer responses
            if response.get("type") in ["query_accepted", "welcome"]:
                continue
            
            # Handle actual answer
            if response.get("status") == "completed":
                answer = response.get("answer", "")
                processing_time = response.get("processing_time", 0)
                
                print(f"\nAnswer ({processing_time:.1f}s):")
                print(f"{answer}")
                return True
                
            elif response.get("status") == "failed":
                error = response.get("error", "Unknown error")
                print(f"\nError: {error}")
                return True
    
    def disconnect(self):
        if self.is_connected:
            try:
                self.socket.send("quit\n".encode('utf-8'))
                self.socket.close()
            except:
                pass
            
            self.is_connected = False
            print("Disconnected")

def simple_interactive_client():
    print("Simple RAG Client")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    client = RAGSocketClient()
    
    # Connect to server
    if not client.connect():
        return
    
    try:
        while True:
            # Get user input
            user_input = input("\nAsk a question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            # Send query
            print("Sending query...")
            if not client.send_query(user_input):
                break
            
            print("Generating answer...")
            
            # Wait for response (blocking)
            if not client.wait_for_answer():
                break
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except EOFError:
        print("\nEOF received")
    finally:
        client.disconnect()

if __name__ == "__main__":
    print("Starting interactive client :)")
    simple_interactive_client()