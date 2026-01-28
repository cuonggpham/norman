import './Header.css';

/**
 * Application header with branding
 * @param {Object} props
 * @param {Function} props.onNewChat - Callback to start a new chat when logo is clicked
 */
export default function Header({ onNewChat }) {
    const handleLogoClick = () => {
        if (onNewChat) {
            onNewChat();
        }
    };

    return (
        <header className="header">
            <div className="header__content">
                <div
                    className="header__brand"
                    onClick={handleLogoClick}
                    style={{ cursor: onNewChat ? 'pointer' : 'default' }}
                    title="Bắt đầu cuộc trò chuyện mới"
                >
                    <span className="header__logo">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 3v18" />
                            <path d="M3 7h18" />
                            <path d="M3 7l3 10h3" />
                            <path d="M15 17h3l3-10" />
                            <circle cx="6" cy="17" r="3" />
                            <circle cx="18" cy="17" r="3" />
                        </svg>
                    </span>
                    <h1 className="header__title">Norman</h1>
                    <span className="header__subtitle">
                        <span className="header__status"></span>
                        Japanese Legal AI
                    </span>
                </div>
            </div>
        </header>
    );
}
