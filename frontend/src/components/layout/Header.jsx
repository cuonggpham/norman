import './Header.css';

/**
 * Application header with branding
 */
export default function Header() {
    return (
        <header className="header">
            <div className="header__content">
                <div className="header__brand">
                    <span className="header__logo">⚖️</span>
                    <h1 className="header__title">Norman</h1>
                    <span className="header__subtitle">Japanese Legal RAG</span>
                </div>
            </div>
        </header>
    );
}
