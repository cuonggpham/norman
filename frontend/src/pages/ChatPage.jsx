import { useRef } from 'react';
import Header from '../components/layout/Header';
import ChatContainer from '../components/chat/ChatContainer';
import './ChatPage.css';

/**
 * Main chat page layout
 */
export default function ChatPage() {
    const chatContainerRef = useRef(null);

    const handleNewChat = () => {
        if (chatContainerRef.current) {
            chatContainerRef.current.resetChat();
        }
    };

    return (
        <div className="chat-page">
            <Header onNewChat={handleNewChat} />
            <main className="chat-page__main">
                <ChatContainer ref={chatContainerRef} />
            </main>
        </div>
    );
}
