import Header from '../components/layout/Header';
import ChatContainer from '../components/chat/ChatContainer';
import './ChatPage.css';

/**
 * Main chat page layout
 */
export default function ChatPage() {
    return (
        <div className="chat-page">
            <Header />
            <main className="chat-page__main">
                <ChatContainer />
            </main>
        </div>
    );
}
