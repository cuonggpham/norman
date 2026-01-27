/**
 * HighlightedText - Renders legal text with highlighted styling
 * Used in SourceCard to indicate text was used in the answer
 */
import './HighlightedText.css';

export default function HighlightedText({ text, isHighlighted = true }) {
    if (!isHighlighted) {
        return <p className="source-text">{text}</p>;
    }

    return (
        <div className="highlighted-text">
            <div className="highlighted-text__indicator">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                </svg>
                <span>Đoạn văn bản được trích dẫn</span>
            </div>
            <p className="highlighted-text__content">{text}</p>
        </div>
    );
}
